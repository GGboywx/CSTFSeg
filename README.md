CSTFSeg: Boundary-Aware Tidal Flat Segmentation Network


<div align="center"> <img src="[Link to your network architecture image or visualization result]" width="800"/> </div>

⚠️ Important Notes (Read Before Use)
Before you proceed, please kindly note the following regarding the code and resources:

Code Version: The code provided in this repository is the research version used to produce the experimental results in the paper. It may differ slightly from a production-ready standard. We are planning to refactor and update the codebase for better usability in the future.

Pre-trained Weights: Due to ongoing commercial projects and patent applications, pre-trained model weights are NOT available for public download at this moment. Users are encouraged to train the model from scratch using the provided dataset and training scripts.

Maintenance: We will try our best to maintain this repository. If you encounter critical bugs, please feel free to open an issue.

📂 Dataset Access
The China Sentinel-2 Tidal Flat (CSTF) dataset constructed in this work is hosted on Google Earth Engine (GEE) to facilitate large-scale access and processing.

Platform: Google Earth Engine

Access Link: [data_download link]

Description: The dataset is pairs of high-resolution Sentinel-2 imagery and pixel-level annotations, covering diverse tidal flat morphologies across China's coastline.Due to project confidentiality requirements, we are currently releasing only a portion of the dataset. The decision on whether to release the full dataset will be made through internal review and approval at a later stage.

Note: You need a valid GEE account to access and download the data.

🛠️ Installation
Requirements
Python 3.8+

PyTorch >= 1.8.0

CUDA >= 11.0

Setup
Bash

# Clone the repository
git clone https://github.com/GGboywx/CSTFSeg.git
cd CSTFSeg

# Install dependencies
pip install -r requirements.txt
🚀 Usage
1. Data Preparation
After downloading the data from GEE, please organize the file structure as follows:

data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
2. Training
To train the CSTFSeg model from scratch:

Bash

python train.py --backbone resnet50 --batch-size 16 --lr 5e-5 --epochs 100 --data-path ./data
Key arguments:

--backbone: Support resnet50, resnet101.

--aux-weight: Weight for the auxiliary edge loss (default: 0.4).

3. Evaluation
To evaluate the model on the test set:

Bash

python val.py --weight-path ./checkpoints/best_model.pth --data-path ./data/val
🧩 Model Architecture
CSTFSeg consists of three key components tailored for coastal features:

Multi-scale Context Encoder: Hybrid CNN-Transformer design to capture both local texture and global context.

Fuzzy Layer: Explicitly models spectral uncertainty in water-sediment transition zones.

Edge-Guided Decoder: Uses auxiliary edge supervision to refine boundary details.

(For detailed mathematical formulations, please refer to Section 3 of our paper.)

📜 Citation
@article{gu5858575cstfseg,
  title={CSTFSeg: A High-Resolution Chinese Tidal Flat Dataset and Multi-Scale Attention Semantic Segmentation Network},
  author={Gu, Wenxuan and Su, Qianqian and Lei, Hui and Shen, Shiqi and Chen, Pengyu and Yu, Zhifeng and Huang, Bei and Wang, Lidong and Zhou, Bin},
  journal={Available at SSRN 5858575}
}
