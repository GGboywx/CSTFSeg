import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from yynet1 import CDNet  # 确保 CDNet 已包含 EdgeGuidedDecoder 的最新修改
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def generate_edge(mask, ksize=3):
    """
    mask: numpy array (H, W), binary mask {0,1}
    ksize: 边界厚度
    return: edge map {0,1}
    """
    # 确保 ksize 为奇数
    if ksize % 2 == 0:
        ksize += 1

    kernel = np.ones((ksize, ksize), np.uint8)
    dil = cv2.dilate(mask.astype(np.uint8), kernel)
    ero = cv2.erode(mask.astype(np.uint8), kernel)
    edge = dil - ero
    return edge


# ========== 原始损失函数 ==========
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # BCE_loss 是带 logits 的，所以不需要 sigmoid
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # p_t = exp(-BCE_loss) is the probability of the true class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        TP = (inputs * targets).sum()
        FN = ((1 - inputs) * targets).sum()
        FP = (inputs * (1 - targets)).sum()
        tversky = (TP + smooth) / (TP + self.alpha * FN + self.beta * FP + smooth)
        return 1 - tversky


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


# ========== 新增: Focal Tversky Loss ==========
class FocalTverskyLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.7, beta=0.3):
        super(FocalTverskyLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # 惩罚假阴性 (FN)
        self.beta = beta  # 惩罚假阳性 (FP)

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        TP = (inputs * targets).sum()
        FN = ((1 - inputs) * targets).sum()
        FP = (inputs * (1 - targets)).sum()
        tversky = (TP + smooth) / (TP + self.alpha * FN + self.beta * FP + smooth)
        focal_term = (1 - tversky) ** self.gamma
        return focal_term


# ========== 数据集定义 ==========
class TideSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, edge_ksize_list=None):
        super(TideSegDataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_list = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        # todo 核心修改1: 新增 edge_ksize_list 参数
        self.edge_ksize_list = edge_ksize_list if edge_ksize_list else [3]  # Default to [3] if not provided

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        image_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace(".jpg", ".png").replace(".jpeg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # 二值化掩码
        if not np.all(np.isin(mask, [0, 1])):
            mask = (mask > 0).astype(np.uint8)

        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # (C, H, W)
            mask = augmented['mask']  # (H, W)

        # 将 mask tensor 转回 numpy 以便生成边缘 (albumentations 的 ToTensorV2 已经将其转换为 tensor)
        # 确保这里是 (H, W) 的 numpy 数组
        mask_np_for_edge = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

        # todo 核心修改2: 生成多尺度的边缘掩码列表
        multi_ksize_edge_masks = []
        for ksize in self.edge_ksize_list:
            edge_mask_np = generate_edge(mask_np_for_edge, ksize=ksize)
            # 每个边缘掩码转换为 (1, H, W) 的 float tensor
            multi_ksize_edge_masks.append(torch.from_numpy(edge_mask_np).float().unsqueeze(0))

        # 主掩码转换为 long type，并确保是 (H, W)
        mask = mask.long()  # For segmentation loss

        # 返回图像，主掩码，以及多尺度边缘掩码列表
        return image, mask, multi_ksize_edge_masks


# ========== 验证函数 ==========
def validate(model, dataloader, device, num_classes, config):
    model.eval()
    total_correct_pixels = 0
    total_num_pixels = 0
    running_val_loss = 0.0

    conf_matrix = torch.zeros(2, 2, dtype=torch.long, device=device)

    bce_criterion = nn.BCEWithLogitsLoss()
    tversky_criterion = TverskyLoss(alpha=config['tversky_alpha'], beta=config['tversky_beta'])
    focal_criterion = FocalLoss(gamma=config['focal_gamma'])
    focal_tversky_criterion = FocalTverskyLoss(gamma=config['focal_tversky_gamma'],
                                               alpha=config['focal_tversky_alpha'],
                                               beta=config['focal_tversky_beta'])


    with torch.no_grad():
        # todo 核心修改3: 接收 multi_ksize_edge_masks
        for images, masks, multi_ksize_edge_masks in tqdm(dataloader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            # 将每个 ksize 的边缘掩码列表转换为一个堆叠的 tensor
            # multi_ksize_edge_masks 是一个列表，每个元素是 (B, 1, H, W)
            # 堆叠后是 (len(ksize_list), B, 1, H, W)
            multi_ksize_edge_masks_tensor = torch.stack(multi_ksize_edge_masks, dim=0).to(device)

            masks_with_channel = masks.unsqueeze(1)  # (B, 1, H, W)

            # todo 更改：接收三个输出，并传递 img_size
            # all_edge_outputs 是一个元组: (edge_out_s4, edge_out_s8, edge_out_s16)
            refined_feat, outputs, all_edge_outputs = model(images, images.shape[2:])

            preds = (torch.sigmoid(outputs) > 0.5).long()

            # todo 更改：计算总损失 = 分割损失 + 边缘损失（考虑多尺度边缘损失）
            seg_loss = 0.0 # 确保 seg_loss 在这里初始化

            if config['loss_function_type'] == 'BCE_Tversky_Focal_Combination':
                loss_bce = bce_criterion(outputs, masks_with_channel.float())
                loss_tversky = tversky_criterion(outputs, masks_with_channel.float())
                loss_focal = focal_criterion(outputs, masks_with_channel.float())
                seg_loss = (config['bce_weight'] * loss_bce) + \
                           (config['tversky_weight'] * loss_tversky) + \
                           (config['focal_weight'] * loss_focal)
            elif config['loss_function_type'] == 'Focal_Tversky_Loss':
                seg_loss = focal_tversky_criterion(outputs, masks_with_channel.float())
            else:
                # 理论上不会走到这里，因为 main 函数会检查 loss_function_type
                raise ValueError(f"Unsupported loss function type: {config['loss_function_type']}")

            # todo 核心修改5: 计算多尺度边缘损失 (健壮性增强)
            total_edge_loss = torch.tensor(0.0, device=device)  # 确保初始化为浮点数

            # 只有当模型实际输出了有效的边缘预测时，才计算边缘损失
            # 检查 all_edge_outputs 的第一个元素是否为 None，即可判断是否启用了 EdgeGuidedDecoder
            if all_edge_outputs[0] is not None:
                if 'multi_scale_edge_loss_weights' in config and config['multi_scale_edge_loss_weights'] is not None \
                        and 'multi_ksize_edge_loss_weights' in config and config[
                    'multi_ksize_edge_loss_weights'] is not None:
                    # 遍历模型所有尺度的边缘输出
                    for i_model_edge, model_edge_output in enumerate(all_edge_outputs):
                        # 对每个模型输出，计算其与所有 ksize 边缘真值的加权损失
                        current_model_edge_loss = 0.0  # 确保初始化为浮点数
                        for i_ksize, ksize_edge_mask in enumerate(multi_ksize_edge_masks_tensor):
                            # ksize_edge_mask 形状为 (B, 1, H, W)
                            current_model_edge_loss += config['multi_ksize_edge_loss_weights'][i_ksize] * \
                                                       bce_criterion(model_edge_output, ksize_edge_mask)

                        # 将该模型尺度的边缘损失，乘以其在 config['multi_scale_edge_loss_weights'] 中的权重
                        total_edge_loss += config['multi_scale_edge_loss_weights'][
                                               i_model_edge] * current_model_edge_loss
                else:
                    # 如果未定义多尺度真值权重，但模型有边缘输出，则退化到只使用第一个模型输出和第一个 ksize 真值
                    total_edge_loss = bce_criterion(all_edge_outputs[0], multi_ksize_edge_masks_tensor[0])
            # else: 如果 all_edge_outputs[0] 为 None，则 total_edge_loss 保持为 0.0

            # 总损失始终计算，与分割损失和边缘损失的计算逻辑平级
            loss = seg_loss + config['aux_weight'] * total_edge_loss # <--- 已修正缩进

            running_val_loss += loss.item()

            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            total_correct_pixels += (preds_flat == masks_flat).sum().item()
            total_num_pixels += masks_flat.numel()

            index = masks_flat * 2 + preds_flat
            counts = torch.bincount(index, minlength=4)
            batch_conf_matrix = counts.reshape(2, 2)
            conf_matrix += batch_conf_matrix

    avg_val_loss = running_val_loss / len(dataloader)
    tp = conf_matrix.diag()
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp

    iou_per_class = torch.where(tp + fp + fn > 0, tp.float() / (tp + fp + fn).float(), torch.tensor(0.0, device=device))
    mean_iou = iou_per_class.mean().item()
    precision_per_class = torch.where(tp + fp > 0, tp.float() / (tp + fp).float(), torch.tensor(0.0, device=device))
    mean_precision = precision_per_class.mean().item()
    recall_per_class = torch.where(tp + fn > 0, tp.float() / (tp + fn).float(), torch.tensor(0.0, device=device))
    mean_recall = recall_per_class.mean().item()
    dice_per_class = torch.where(2 * tp + fp + fn > 0, (2 * tp).float() / (2 * tp + fp + fn).float(),
                                 torch.tensor(0.0, device=device))
    mean_dice = dice_per_class.mean().item()
    avg_acc = total_correct_pixels / total_num_pixels
    model.train()

    print("\n--- Validation Metrics ---")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Class 0 (Background) IoU: {iou_per_class[0].item():.4f}")
    print(f"Class 1 (Tidal Flat) IoU: {iou_per_class[1].item():.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Overall Accuracy: {avg_acc:.4f}")

    return avg_val_loss, mean_iou, mean_dice, mean_precision, mean_recall, avg_acc


# ========== 训练函数 ==========
def train_one_epoch(model, dataloader, optimizer, device, epoch, writer, config):
    model.train()
    running_loss = 0.0
    running_seg_loss = 0.0  # 记录分割损失
    running_edge_loss = 0.0  # 记录边缘损失

    bce_criterion = nn.BCEWithLogitsLoss()
    tversky_criterion = TverskyLoss(alpha=config['tversky_alpha'], beta=config['tversky_beta'])
    focal_criterion = FocalLoss(gamma=config['focal_gamma'])
    focal_tversky_criterion = FocalTverskyLoss(gamma=config['focal_tversky_gamma'],
                                               alpha=config['focal_tversky_alpha'],
                                               beta=config['focal_tversky_beta'])

    # todo 核心修改4: 接收 multi_ksize_edge_masks
    for i, (images, masks, multi_ksize_edge_masks) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        images, masks = images.to(device), masks.to(device)
        # 将每个 ksize 的边缘掩码列表转换为一个堆叠的 tensor
        multi_ksize_edge_masks_tensor = torch.stack(multi_ksize_edge_masks, dim=0).to(device)  # (num_ksize, B, 1, H, W)

        optimizer.zero_grad()

        # todo 更改：接收三个输出，并传递 img_size
        # all_edge_outputs 是一个元组: (edge_out_s4, edge_out_s8, edge_out_s16)
        refined_feat, outputs, all_edge_outputs = model(images, images.shape[2:])
        masks_with_channel = masks.unsqueeze(1)

        # 计算主分割损失
        seg_loss = 0.0 # 确保 seg_loss 在这里初始化

        if config['loss_function_type'] == 'BCE_Tversky_Focal_Combination':
            loss_main_bce = bce_criterion(outputs, masks_with_channel.float())
            loss_main_tversky = tversky_criterion(outputs, masks_with_channel.float())
            loss_main_focal = focal_criterion(outputs, masks_with_channel.float())
            seg_loss = (config['bce_weight'] * loss_main_bce) + \
                       (config['tversky_weight'] * loss_main_tversky) + \
                       (config['focal_weight'] * loss_main_focal)
        elif config['loss_function_type'] == 'Focal_Tversky_Loss':
            seg_loss = focal_tversky_criterion(outputs, masks_with_channel.float())
        else:
            # 理论上不会走到这里，因为 main 函数会检查 loss_function_type
            raise ValueError(f"Unsupported loss function type: {config['loss_function_type']}")


        # todo 核心修改5: 计算多尺度边缘损失
        total_edge_loss = torch.tensor(0.0, device=device)  # 确保初始化为浮点数，且在最外层

        # 只有当模型实际输出了有效的边缘预测时，才计算边缘损失
        if all_edge_outputs[0] is not None:
            if 'multi_scale_edge_loss_weights' in config and config['multi_scale_edge_loss_weights'] is not None \
                    and 'multi_ksize_edge_loss_weights' in config and config[
                'multi_ksize_edge_loss_weights'] is not None:
                # 遍历模型所有尺度的边缘输出
                for i_model_edge, model_edge_output in enumerate(all_edge_outputs):
                    # 对每个模型输出，计算其与所有 ksize 边缘真值的加权损失
                    current_model_edge_loss = 0.0  # 确保初始化为浮点数
                    for i_ksize, ksize_edge_mask in enumerate(multi_ksize_edge_masks_tensor):
                        # ksize_edge_mask 形状为 (B, 1, H, W)
                        current_model_edge_loss += config['multi_ksize_edge_loss_weights'][i_ksize] * \
                                                   bce_criterion(model_edge_output, ksize_edge_mask)

                    # 将该模型尺度的边缘损失，乘以其在 config['multi_scale_edge_loss_weights'] 中的权重
                    total_edge_loss += config['multi_scale_edge_loss_weights'][i_model_edge] * current_model_edge_loss
            else:
                # 如果未定义多尺度真值权重，但模型有边缘输出，则退化到只使用第一个模型输出和第一个 ksize 真值
                total_edge_loss = bce_criterion(all_edge_outputs[0], multi_ksize_edge_masks_tensor[0])
        # else: 如果 all_edge_outputs[0] 为 None，则 total_edge_loss 保持为 0.0

        # todo 更改：总损失 = 分割损失 + 边缘损失
        loss = seg_loss + config['aux_weight'] * total_edge_loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_seg_loss += seg_loss.item()
        running_edge_loss += total_edge_loss.item()  # 记录加权后的总边缘损失

        if writer:
            global_step = epoch * len(dataloader) + i
            writer.add_scalar('Train/Total_Loss', loss.item(), global_step)
            writer.add_scalar('Train/Seg_Loss', seg_loss.item(), global_step)
            writer.add_scalar('Train/Edge_Loss', total_edge_loss.item(), global_step)  # 记录加权后的总边缘损失

            if global_step % 100 == 0:
                with torch.no_grad():
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                    grid_images = make_grid(images[:4], nrow=2, normalize=True)
                    grid_preds = make_grid(preds[:4], nrow=2)
                    grid_masks = make_grid(masks.unsqueeze(1)[:4].float(), nrow=2)

                    writer.add_image('Train/Images', grid_images, global_step)
                    writer.add_image('Train/Predictions', grid_preds, global_step)
                    writer.add_image('Train/GroundTruth', grid_masks, global_step)

                    # 只有当模型有边缘预测输出时，才记录边缘相关的图像
                    if all_edge_outputs[0] is not None:
                        edge_preds_to_display = (torch.sigmoid(all_edge_outputs[0]) > 0.5).float()
                        edge_masks_to_display = multi_ksize_edge_masks_tensor[0]  # 边缘真值总是存在的

                        grid_edge_preds = make_grid(edge_preds_to_display[:4], nrow=2)
                        grid_edge_masks = make_grid(edge_masks_to_display[:4], nrow=2)

                        writer.add_image('Train/Edge_Predictions', grid_edge_preds, global_step)
                        writer.add_image('Train/Edge_GroundTruth', grid_edge_masks, global_step)
                    else:
                        # 如果没有边缘预测，可以记录一个空白图像或者不记录
                        pass  # 或者可以考虑添加一个占位符图像，例如全黑图像

    return running_loss / len(dataloader)


# ========== 主函数入口 ==========
def main():
    config = {
        # 数据路径
        'image_dir': '/home/GWX/FTSAM/data/td_final/train/image',
        'mask_dir': '/home/GWX/FTSAM/data/td_final/train/mask',
        'val_image_dir': '/home/GWX/FTSAM/data/td_final/val/image',
        'val_mask_dir': '/home/GWX/FTSAM/data/td_final/val/mask',
        'save_dir': '/home/GWX/FTSAM/output',
        'log_dir_base': './logs',

        # 数据加载参数
        'batch_size': 8,
        'val_batch_size': 8,
        'num_workers': 4,

        # 模型参数
        'backbone': 'resnet50',
        'output_stride': 16,
        'img_size': 1024,
        'num_classes': 1,
        'img_chan': 3,
        'chan_num': 64,

        # todo 核心修改：在 train.py 中设置 CDNet 模块开关
        # 这些是针对 Exp. 1.1 BL+BackboneEnhance 的配置
        'use_aspp_module': True,
        'use_mft_transformer': False,
        'use_mft_pe': True,
        'use_edge_guided_decoder': True, # <--- 确保这里设置正确，如果你要运行不带边缘引导解码器的版本
        'use_fuzzy_layer': True,
        'use_cross_edge_fusion': False,
        # 训练参数
        'num_epochs': 400,
        'learning_rate': 5e-5,
        'optimizer_type': 'AdamW',
        'weight_decay': 5e-4,
        'momentum': 0.9,

        # 损失函数配置
        'loss_function_type': 'BCE_Tversky_Focal_Combination',  # 或 'Focal_Tversky_Loss'
        'bce_weight': 0.5,
        'tversky_weight': 0.3,
        'focal_weight': 0.2,
        'focal_gamma': 2.0,

        'aux_weight': 0.45,  # 边缘损失的总权重 (应用于总的加权边缘损失)
        'tversky_alpha': 0.3,
        'tversky_beta': 0.7,
        'focal_tversky_gamma': 2.0,
        'focal_tversky_alpha': 0.8,
        'focal_tversky_beta': 0.2,

        'fuzzy_num': 16,

        # todo 核心修改6: 新增边缘粗细度和其权重配置
        'edge_ksize_list': [3, 5, 7],  # 定义多尺度边缘的粗细度，generate_edge 的 ksize
        # 对应 EdgeGuidedDecoder 返回的 (s4, s8, s16) 顺序的模型边缘预测头的权重
        'multi_scale_edge_loss_weights': [0.5, 0.3, 0.2],
        # 对应 edge_ksize_list [3, 5, 7] 顺序的边缘真值损失权重，总和建议为 1.0
        'multi_ksize_edge_loss_weights': [0.5, 0.3, 0.2],

        # 学习率调度器
        'use_lr_scheduler': True,
        'lr_scheduler_type': 'ReduceLROnPlateau',
        'lr_patience': 10,
        'lr_factor': 0.5,
        't_max': 500,

        # 早停
        'use_early_stopping': True,
        'patience': 30,
        'min_delta': 0.0001,

        # 归一化参数,**已经确定！**
        'mean': (0.3876, 0.4297, 0.4462),
        'std': (0.2091, 0.1981, 0.1846),

        # Albumentations 增强参数,**已经确定！**
        'aug_horizontal_flip': 0.5,
        'aug_vertical_flip': 0.5,
        'aug_random_rotate_90': 0.5,
        'aug_shift_scale_limit': 0.0625,
        'aug_shift_scale_p': 0.7,
        'aug_random_scale_limit': (0.8, 1.2),
        'aug_random_scale_p': 0.8,
        'aug_brightness_contrast_limit': 0.2,
        'aug_brightness_contrast_p': 0.7,
        'aug_hue_saturation_value_limit': 0.2,
        'aug_hue_saturation_value_p': 0.7,
        'aug_gauss_noise_limit': (10, 50),
        'aug_gauss_noise_p': 0.3,
        'aug_coarse_dropout_p': 0.2,
        'aug_coarse_dropout_max_holes': 8,
        'aug_coarse_dropout_max_height': 64,
        'aug_coarse_dropout_max_width': 64,
        # 新增形变增强
        'aug_elastic_transform_p': 0.3,

        'aug_grid_distortion_p': 0.3,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(config['save_dir'], exist_ok=True)

    # ==== Albumentations 数据增强管道 ====
    train_transform = A.Compose([
        A.HorizontalFlip(p=config['aug_horizontal_flip']),
        A.VerticalFlip(p=config['aug_vertical_flip']),
        A.RandomRotate90(p=config['aug_random_rotate_90']),
        # ShiftScaleRotate 应该在 PadIfNeeded 之前，避免裁剪问题
        A.ShiftScaleRotate(
            shift_limit=config['aug_shift_scale_limit'],
            scale_limit=0,  # Scale handled by RandomScale
            rotate_limit=0,  # Rotate handled by RandomRotate90
            p=config['aug_shift_scale_p'],
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
        ),
        A.RandomScale(scale_limit=config['aug_random_scale_limit'], p=config['aug_random_scale_p']),
        A.ColorJitter(
            brightness=config['aug_brightness_contrast_limit'],
            contrast=config['aug_brightness_contrast_limit'],
            p=config['aug_brightness_contrast_p']
        ),
        A.HueSaturationValue(
            hue_shift_limit=config['aug_hue_saturation_value_limit'],
            sat_shift_limit=config['aug_hue_saturation_value_limit'],
            val_shift_limit=config['aug_hue_saturation_value_limit'],
            p=config['aug_hue_saturation_value_p']
        ),
        A.GaussNoise(var_limit=config['aug_gauss_noise_limit'], p=config['aug_gauss_noise_p']),
        A.CoarseDropout(
            max_holes=config['aug_coarse_dropout_max_holes'],
            max_height=config['aug_coarse_dropout_max_height'],
            max_width=config['aug_coarse_dropout_max_width'],
            fill_value=0,
            mask_fill_value=0,
            p=config['aug_coarse_dropout_p']
        ),
        # 新增形变增强
        A.ElasticTransform(p=config['aug_elastic_transform_p'], alpha=120, sigma=120 * 0.09, alpha_affine=120 * 0.09,
                           border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.GridDistortion(p=config['aug_grid_distortion_p'], border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.PadIfNeeded(
            min_height=config['img_size'],
            min_width=config['img_size'],
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0
        ),
        A.RandomCrop(height=config['img_size'], width=config['img_size'], p=1.0),
        A.Normalize(
            mean=config['mean'],
            std=config['std'],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=config['img_size'], width=config['img_size']),
        A.Normalize(
            mean=config['mean'],
            std=config['std'],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    # ==== 加载数据 ====
    # todo 核心修改7: 在加载数据集时传递 edge_ksize_list
    train_dataset = TideSegDataset(config['image_dir'], config['mask_dir'],
                                   transform=train_transform,
                                   edge_ksize_list=config['edge_ksize_list'])
    val_dataset = TideSegDataset(config['val_image_dir'], config['val_mask_dir'],
                                 transform=val_transform,
                                 edge_ksize_list=config['edge_ksize_list'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['val_batch_size'], shuffle=False,
                            num_workers=config['num_workers'], pin_memory=True)

    # ==== 构建模型 ====
    model = CDNet(backbone=config['backbone'],
                  output_stride=config['output_stride'],
                  img_size=config['img_size'],
                  n_class=config['num_classes'],
                  img_chan=config['img_chan'],
                  chan_num=config['chan_num'],
                  fuzzy_num=config['fuzzy_num'],
                  # todo 核心修改：将配置中的开关传递给 CDNet
                  use_aspp_module=config['use_aspp_module'],
                  use_mft_transformer=config['use_mft_transformer'],
                  use_mft_pe=config['use_mft_pe'],
                  use_edge_guided_decoder=config['use_edge_guided_decoder'],
                  use_fuzzy_layer=config['use_fuzzy_layer'],
                  use_cross_edge_fusion=config['use_cross_edge_fusion']
                  )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
    model.to(device)


    # ==== 优化器 ====
    optimizer = None
    if config['optimizer_type'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer_type'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer_type'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'],
                              weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer type: {config['optimizer_type']}")

    # ==== 学习率调度器 ====
    scheduler = None
    if config['use_lr_scheduler']:
        if config['lr_scheduler_type'] == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=config['lr_factor'],
                                                             patience=config['lr_patience'], verbose=True)
        elif config['lr_scheduler_type'] == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['t_max'])
        else:
            raise ValueError(f"Unsupported LR scheduler type: {config['lr_scheduler_type']}")

    # ==== TensorBoard ====
    log_dir = os.path.join(config['log_dir_base'], datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # ==== 训练循环 ====
    best_iou = -1.0
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    val_ious = []
    val_dices = []
    val_precisions = []
    val_recalls = []
    val_accs = []

    for epoch in range(config['num_epochs']):
        print(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")
        writer.add_scalar('LearningRate', current_lr, epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer, config)
        val_loss, val_iou, val_dice, val_precision, val_recall, val_acc = validate(model, val_loader, device,
                                                                                   config['num_classes'], config)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_dices.append(val_dice)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1} Summary | Train Loss: {train_loss:.4f}")
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/IoU", val_iou, epoch)
        writer.add_scalar("Val/Dice", val_dice, epoch)
        writer.add_scalar("Val/Precision", val_precision, epoch)
        writer.add_scalar("Val/Recall", val_recall, epoch)
        writer.add_scalar("Val/Acc", val_acc, epoch)

        if config['use_lr_scheduler']:
            if config['lr_scheduler_type'] == 'ReduceLROnPlateau':
                scheduler.step(val_iou)
            elif config['lr_scheduler_type'] == 'CosineAnnealingLR':
                scheduler.step()

        if val_iou > best_iou + config['min_delta']:
            best_iou = val_iou
            best_epoch = epoch + 1
            best_dice = val_dice
            best_precision = val_precision
            best_recall = val_recall
            best_acc = val_acc
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                       os.path.join(config['save_dir'], f'xr_model_{config["backbone"]}.pth'))
            print(f"✅ 最佳模型已更新并保存于轮次 {epoch + 1}, 验证 Mean IoU={best_iou:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"模型未改善 Mean IoU. 连续 {epochs_no_improve}/{config['patience']} 次未改善.")
            if config['use_early_stopping'] and epochs_no_improve >= config['patience']:
                print(f"\n🚫 早停触发! 验证 Mean IoU 连续 {config['patience']} 次未改善。")
                print(f"🚀 最佳模型信息:")
                print(f"   最佳轮次: {best_epoch}")
                print(f"   最佳 Mean IoU: {best_iou:.4f}")
                print(f"   最佳 Mean Dice: {best_dice:.4f}")
                print(f"   最佳 Mean Precision: {best_precision:.4f}")
                print(f"   最佳 Mean Recall: {best_recall:.4f}")
                print(f"   最佳 Overall Accuracy: {best_acc:.4f}")
                break

    writer.close()
    print("训练完成！")

    # ========== 绘图保存 ==========
    epochs_trained = len(train_losses)
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs_trained), train_losses, label="Train Loss")
    plt.plot(range(epochs_trained), val_losses, label="Val Loss", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], "loss_curves.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs_trained), val_ious, label="Val Mean IoU", color='darkgreen')
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Validation Mean IoU Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], "iou_curve.png"))
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs_trained), val_dices, label="Val Mean Dice", color='purple')
    plt.plot(range(epochs_trained), val_precisions, label="Val Mean Precision", color='orange', linestyle=':')
    plt.plot(range(epochs_trained), val_recalls, label="Val Mean Recall", color='red', linestyle='-.')
    plt.plot(range(epochs_trained), val_accs, label="Val Overall Accuracy", color='skyblue', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics (Dice, Precision, Recall, Accuracy)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], "other_metrics_curves.png"))
    plt.close()

    print(f"所有训练曲线图已保存至: {config['save_dir']}")


if __name__ == '__main__':
    main()