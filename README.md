Self-Supervised Denoising of LODOX® Statscan® Images
====================================================

This repository contains the complete code and experimental notebooks for the B.Sc. thesis: _"Self-supervised Denoising of Images from a LODOX® Statscan® with a Photon Counting Detector"_.

Project Overview
----------------

This project investigates methods to enhance the image quality of low-dose LODOX® medical images, which suffer from high levels of stochastic (quantum) noise that can impede diagnostic accuracy.

A primary, unexpected challenge encountered during data acquisition was the presence of **severe, periodic stripe artifacts** due to a scanner malfunction. The project, therefore, adopted a two-stage approach:

1.  **Stage 1: Artifact Removal (Destriping)**A comparative analysis to find the most effective method for removing the severe stripe artifacts _before_ any denoising could be attempted.
    
2.  **Stage 2: Stochastic Denoising**A comparative analysis of modern self-supervised denoising models (which require no clean reference images) applied to the _artifact-cleaned_ data.
    

Key Findings
------------

1.  **Artifact Removal (Destriping):**The classical frequency-domain algorithm, **Non-Recoverable Compressed Sensing (NRCS)**, was found to be decisively superior for removing the specific stripe artifacts. It successfully removed all artifacts while enhancing overall image contrast. The untrained Deep Image Prior (DIP) network, while flexible, struggled to isolate the stripe pattern and led to oversmoothing and residual artifacts.
    
2.  The models adopted distinct behaviors under this constraint:
    
    *   **Noise2Self (N2S)** tended toward destructive oversmoothing, removing texture.
        
    *   **Deep Image Prior (DIP)** was unable to learn the specific noise pattern effectively.
        
    *   **Neighbor2Neighbor (Nei2Nei)** proved to be the most stable, adopting a conservative "identity function" (i.e., "doing no harm"), which made it the most optimal, albeit non-impactful, choice.
        

**Conclusion:** This work provides a robust, validated pipeline for LODOX® **artifact removal** (NRCS) and a critical demonstration of the **severe data-dependency** of self-supervised _denoisers_, highlighting their specific behaviors under data-scarce conditions.

Repository Structure
--------------------

`Denoising_Lodox_Images/
│
├── Pre-Processing/
│   ├── Creating_Augmented_Dataset.ipynb  # Notebook for data augmentation
│   └── interactive_cropper.py            # Python script for manual ROI cropping
│
├── NRCS/
│   ├── Removing_Artifacts.ipynb          # Core notebook for Stage 1 (NRCS destriping)
│   ├── destripe.py                     # Helper functions for NRCS
│   └── FFTW.py                         # FFT helper functions
│
├── DIP/
│   ├── DeepImagePrior.ipynb              # Core notebook for DIP (used in Stage 1 & 2)
│   ├── models/                           # DIP model architectures (e.g., skip)
│   └── utils/                            # Helper functions for DIP
│
├── N2S/
│   ├── Noise2Self.ipynb                  # Core notebook for Stage 2 (N2S denoising)
│   ├── models/                           # N2S model architectures (U-Net, DnCNN)
│   └── ...                               # Training and utility scripts
│
├── Nei2Nei/
│   ├── Neighbor2Neighbor.ipynb           # Core notebook for Stage 2 (Nei2Nei denoising)
│   └── ...                               # Training and utility scripts
│
├── requirements.txt                      # All Python dependencies
└── README.md                           # This file`

## Getting Started
---------------

### Prerequisites

*   Python 3.9+
    
*   Jupyter Lab or Jupyter Notebook
    
*   A Python virtual environment (e.g., venv or conda) is highly recommended.
    

### Installation

1.  git clone \[https://github.com/your-username/Denoising\_Lodox\_Images.git\](https://github.com/your-username/Denoising\_Lodox\_Images.git)cd Denoising\_Lodox\_Images
    
2.  python -m venv venvsource venv/bin/activate # On Windows, use \`venv\\Scripts\\activate\`
    
3.  pip install -r requirements.txt
    

How to Run the Experiments
--------------------------

The core of this project is contained within the Jupyter Notebooks (.ipynb) in each directory. The experiments and analysis can be reproduced by running the notebooks in the following order:

1.  **Data Preparation (Pre-Processing/)**
    
    *   Place your raw LODOX® images (in .png or .dcm format) into a source folder (e.g., data/raw/).
        
    *   Run Pre-Processing/interactive\_cropper.py to manually define and save the cropping coordinates for your images.
        
    *   Run Pre-Processing/Creating\_Augmented\_Dataset.ipynb to apply the cropping, resize images, and generate the augmented training/validation sets.
        
2.  **Stage 1: Artifact Removal (NRCS/ & DIP/)**
    
    *   Run NRCS/Removing\_Artifacts.ipynb to apply and evaluate the NRCS algorithm on your corrupted images.
        
    *   Run DIP/DeepImagePrior.ipynb and set it to "destripe" mode to compare the results.
        
3.  **Stage 2: Denoising (N2S/, Nei2Nei/, DIP/)**
    
    *   Run the notebooks for each model to train and evaluate them on the _cleaned_ dataset created in Stage 1.
        
    *   N2S/Noise2Self.ipynb
        
    *   Nei2Nei/Neighbor2Neighbor.ipynb
        
    *   DIP/DeepImagePrior.ipynb (set to "denoise" mode)
        

Acknowledgements
----------------

This work was submitted in partial fulfillment of the requirements for the degree of Bachelor of Science in Mechatronic Engineering at the University of Cape Town.

Special thanks to Dr. Yaaseen Martin, Dr. Lindie Du Plessis, and Onke for their invaluable supervision, support, and technical assistance throughout this project.