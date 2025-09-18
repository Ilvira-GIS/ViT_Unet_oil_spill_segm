# Oil Spill Segmentation in SAR Data using ViT-UNet: Performance and Practical Insights

This repository contains the implementation of a Vision Transformer-based U-Net (ViT-UNet) model for segmenting oil spills in Synthetic Aperture Radar (SAR) data. The project focuses on applying deep learning techniques to detect and delineate oil spills in satellite imagery. The code includes the model architecture, training and evaluation scripts, and visualization of the results.

## Dataset Information

The dataset used in this project is a refined version of the Deep-SAR Oil Spill (SOS) dataset, which consists of SAR images annotated for oil spill regions. The original dataset is sourced from Zhu et al. (2021), available via their GitHub repository: https://github.com/CUG-URS/CBDNet-main. Download links for the original dataset:

- Baidu Drive (extraction code: urs6): http://cugurs5477.mikecrm.com/QaXx0Rw
- Google Drive: http://cugurs5477.mikecrm.com/5tk5gyO

The refined version, with manually corrected annotations, is available on Zenodo: https://zenodo.org/records/17012275 (DOI: 10.5281/zenodo.17012275).

## Code Information

The repository contains two main files for the implementation:

1.  **`main.py`**: A fully executable Python script that runs the entire reproducible pipeline from data loading and training to evaluation. This script is designed for command-line execution and guarantees consistent results by setting a random seed.
2.  **`vit_unet_oilspill.ipynb`**: A Jupyter Notebook that provides a more interactive and visual walkthrough of the code. It contains the same core logic as `main.py` but is structured in cells, making it ideal for exploration, debugging, and visualizing results, including basic visualizations of segmentation outputs.

Both files include:
- **Custom Classes:** `ViTEncoder`, `ViTUNet`, `CombinedLoss`.
- **Helper Functions** for model loading, metric computation, and visualization.

## Usage Instructions

1.  **Clone the repository.**

2.  **Set up the environment:**
    - Install Anaconda/Miniconda if needed.
    - Create and activate the environment: `conda create -n vit_unet python=3.9 && conda activate vit_unet`.
    - Install libraries: `pip install torch torchvision torchaudio opencv-python-headless matplotlib pandas albumentations segmentation-models-pytorch timm jupyter`.

3.  **Prepare the dataset:**
    - Download the refined SOS dataset from https://zenodo.org/records/17012275.
    - Extract it into a folder named `dataset`, ensuring the folder structure matches `dataset/images/train`, `dataset/masks/train`, etc.
    - In the `main.py` script or `vit_unet_oilspill.ipynb` notebook, update the `dataset_dir` variable to the correct path of your `dataset` folder.

4.  **Run the code:**
    You have two options to run the code:

    **Option A: Run the Python script (Recommended for reproducibility)**
    - Execute the main script from your terminal: `python main.py`.
    - This will run the full, reproducible pipeline and print the final metrics.

    **Option B: Use the Jupyter Notebook (Recommended for exploration and visualization)**
    - Launch Jupyter: `jupyter notebook`.
    - Open `vit_unet_oilspill.ipynb`.
    - Run the cells sequentially to train the model and see the outputs, including the basic visualizations.

## Requirements

- Python >= 3.8
- PyTorch
- Albumentations
- OpenCV (`opencv-python-headless`)
- Segmentation Models PyTorch (`segmentation-models-pytorch`)
- Timm
- Jupyter

## Methodology

- **Data Preprocessing**
  - All images are resized to 224x224 using bilinear interpolation.
  - Training data is augmented with horizontal/vertical flips (p=0.5) and random 90° rotations (p=0.5).
  - The dataset is split into 6,455 training images and 1,615 validation images.

- **Evaluation Method**
  - **Ablation Study:** The model's performance was compared on the original vs. the refined dataset to demonstrate the impact of annotation quality.
  - **Final Model Performance:** The final model is trained on the training set and evaluated on the validation set. Metrics are calculated excluding images with all-zero (empty) ground truth masks to focus on the model's ability to segment actual spills.

- **Assessment Metrics**
  - **mean Intersection over Union (mIoU):** Measures the overlap between predicted and true masks. It is a standard metric for segmentation that penalizes both false positives and false negatives.
  - **F1-Score:** The harmonic mean of Precision and Recall, providing a balanced measure, which is especially useful for imbalanced datasets like oil spill detection.
  - **Precision:** Measures the accuracy of positive predictions, minimizing false positives (i.e., classifying non-spill areas as spills).
  - **Recall:** Measures the model's ability to find all actual spills, minimizing false negatives.
  - **Accuracy:** Measures the overall pixel-wise correctness.

## Citations

- **Original SOS Dataset:** Zhu, Q., Zhang, Y., Li, Z., Yan, X., Guan, Q., Zhong, Y., Zhang, L., and Li, D. (2021). Oil spill contextual and boundary-supervised detection network based on marine sar images. *IEEE Transactions on Geoscience and Remote Sensing, 60*:1–10.
- **Refined SOS Dataset:** Anonymous authors (2025). Refined deep-sar oil spill (sos) dataset. [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17012275

## License

This project is licensed under the MIT License.
