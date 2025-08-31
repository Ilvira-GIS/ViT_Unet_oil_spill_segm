# Oil Spill Segmentation in SAR Data using ViT-UNet: Performance and Practical Insights

This repository contains the implementation of a Vision Transformer-based U-Net (ViT-UNet) model for segmenting oil spills in Synthetic Aperture Radar (SAR) data. The project focuses on applying deep learning techniques to detect and delineate oil spills in satellite imagery. The code includes model architecture, training scripts, evaluation metrics, and visualization of the results obtained.

**Dataset**

The dataset used in this project is a refined version of the Deep-SAR Oil Spill (SOS) dataset, which consists of SAR images annotated for oil spill regions. The original dataset is sourced from Zhu et al. (2021), available via their GitHub repository: https://github.com/CUG-URS/CBDNet-main. Download links for the original dataset:

Baidu Drive (extraction code: urs6)：http://cugurs5477.mikecrm.com/QaXx0Rw

Google Drive：http://cugurs5477.mikecrm.com/5tk5gyO

The refined version, with manually corrected annotations, is available on Zenodo: https://zenodo.org/records/15298010 (DOI: 10.5281/zenodo.15298010).

**Code Information**

The implementation is provided in the Jupyter notebook vit_unet_oilspill.ipynb, which includes:

*Custom classes:*

class ViTEncoder(nn.Module): Wraps a pre-trained Vision Transformer (ViT) from timm as the encoder to extract global features from input images, processing them into patch embeddings with self-attention mechanisms. It uses ViT-Small configuration (patch size 16, embed dim 384) and provides skip connections from blocks 2, 5, and 8.

class ViTUNet(nn.Module): The full segmentation model combining the ViTEncoder with a U-Net-style decoder. The decoder upsamples features using transposed convolutions and integrates skip connections for precise spatial reconstruction, outputting binary segmentation masks.

class CombinedLoss(nn.Module): Implements the hybrid loss function as a weighted combination of Binary Cross-Entropy (BCE) loss (weight 0.8) for pixel-wise classification and Dice loss (weight 0.2) to handle class imbalance and improve boundary accuracy.


*Custom functions* for model loading (load_model), metric computation (compute_metrics for mIoU, F1, Precision, Recall, Accuracy), contour extraction (get_contours), image loading (load_original_image), and visualization (visualize_segmentation for overlaying ground truth and predicted contours).

Main execution for loading the pre-trained model (vit_unet_oil_segmentation.pth), evaluating on validation data, and visualizing results.

**Usage Instructions** 

1. Download the repository from
   
2. Set up environment:
   
Install Anaconda/Miniconda if needed.

Create/activate env: conda create -n vit_unet python=3.9 && conda activate vit_unet.

Install libs: pip install -r requirements.txt

3. Prepare dataset:

Download refined SOS from https://zenodo.org/records/15298010.

Extract to './dataset' and properly specify the path "dataset_dir = r'C:/Users/Username/dataset'"
 
4. Run the notebook:

Launch Jupyter: jupyter notebook.

Open vit_unet_oilspill.ipynb.

For full reproduction (including model training), run all cells consecutively.

**Requirements**

- Python >= 3.8

- PyTorch

- Albumentations

- OpenCV

- Segmentation Models PyTorch (segmentation-models-pytorch)

- Timm

**Methodology**

*Data Preprocessing*

Resize to 224x224 via bilinear interpolation.

Augment training: Horizontal/vertical flips (p=0.5), random 90° rotations (p=0.5).

Split: 6,455 train, 1,615 validation (used for evaluation).

*Modeling*

Architecture: ViT-UNet with pre-trained ViT-Small encoder for global features; U-Net decoder upsamples with skip connections.

Training: AdamW optimizer (lr=1e-5, weight_decay=1e-2), hybrid loss (0.8 BCE + 0.2 Dice). ReduceLROnPlateau scheduler. Batch size 4, 10 epochs.

Evaluation: Ablation on original vs. refined data; final on validation, excluding all-zero masks. Threshold 0.5.

*Metrics*

mIoU: Overlap, penalizes FP/FN.

F1: Balances precision/recall for imbalance.

Precision: Minimizes false positives.

Recall: Detects all spills.

Accuracy: Pixel correctness.

**Citations**

- Original SOS Dataset: Zhu, Q., Zhang, Y., Li, Z., Yan, X., Guan, Q., Zhong, Y., Zhang, L., and Li, D. (2021). Oil spill contextual and boundary-supervised detection network based on marine sar images. IEEE Transactions on Geoscience and Remote Sensing, 60:1–10.

- Refined SOS Dataset: Anonimous authors (2025). Refined deep-sar oil spill (sos) dataset. [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15298010
