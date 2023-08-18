[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Data for the Empirical Analysis of the Paper [A Deep Learning and Image ProcessingPipeline for Object Characterization in Firm Operations].

This archive is distributed in association with the [INFORMS Journal on Computing] under the [MIT License](LICENSE).

This repository includes the data used in the empirical analysis of the paper
[A Deep Learning and Image Processing Pipeline for Object Characterization in Firm Operations] by A. Aghasi and
A. Rai and Y. Xia.

## Cite

To cite this software, please cite the paper using its DOI and the software itself, using the
following DOI.

Below is the BibTex for citing this version of the repository.
```
@article{ObjCh,
author = { A. Aghasi and A. Rai and Y. Xia },
publisher = {INFORMS Journal on Computing},
title = {{ObjCh} Version v2022.0260},
year = {2022},
doi = {10.5281/zenodo.7348935},
url = {https://github.com/INFORMSJoC/2022.0260},
}

```
--------------------------------------------------------------------------
## Code and Datasets:

1.Large Image to Small Batch
  - Code:
    -  Image_chunk_make.ipynb: The code is desigined for splitting the large images into smaller pathes to create larger training datasets.The output would be a bunch of patches with the size of 256 X 256.
  - Sample Images:
    -  cardbord/ contains 10% of the datasets used in the paper due to the limitation of the size of all files. Train folder contains image and label sub-folder with and small patches for model training. test folder contains small patches for prediction. raw data folder contains all original images. predicted folder contains the predicted images.
  - Complete dataset in google drive: (https://drive.google.com/drive/folders/1vuGQWBd6DpS9gpwWTFXdlrznNlIselyl?usp=drive_link)
2. Module: Unet 
  - Code:
    -  Unet_model.ipynb: methods for the unet model training to identify the select object with the ground truth/masks. The output would be binary object with removed background.
  - Sample Images:
    -  Woodlog/ contains 10% of the datasets used in the paper due to the limitation of the size of all files. Train folder contains image and label sub-folder with and small patches for model training. test folder contains small patches for prediction. raw data folder contains all original images. predicted folder contains the predicted images.
  - Complete dataset in google drive: (https://drive.google.com/drive/folders/1nWVjByuTQbuWfbX6BUn9KAf4i_I3s83T?usp=sharing)
3. MUSIC:
  - Code:
    - MUSICCountLayers.m: used to implement the MUSIC algorithm and extract an estimate of layer counts.
    - binaryLogCount.m: provides the code to characterize and count the number of log cross-sections.
  - Images:
    - MUSIC/logbinary: woodlog predicted iamges for counting the number of log cross-sections.
    - MUSIC/cardboard: cardboard predicted images for estimating the number of layer.
4. comparison between UNET and other ML approaches:
  - Code:
    - ProcessImages.m: convert images into strips to be used for comparison between UNET and other machine learning approaches.
    - M1.py: Linear regression (input: color, vectorized)
    - M2.py: Lasso regression (input: color, vectorized)
    - M3.py: Principal Component Regression (input: color, vectorized)
    - M4.py: Random forests (input: color, vectorized)
    - M5.py: ADA-Boost (input: color, vectorized)
    - M6.py: DNN: 2 conv layers-2 FC layers (input: color, 3D tensor)
    - M7.py: DNN: 4 conv layers- 3 FC layers (input: color, 3D tensor)
    - M8.py: Linear regression (input: grayscale, vectorized)
    - M9.py: Lasso regression (input: grayscale, vectorized)
    - M10.py: Principal Component Regression (input: grayscale, vectorized)
    - M11.py: Random forests (input: grayscale, vectorized)
    - M12.py: ADA-Boost (input: grayscale, vectorized)
    - M13.py: DNN: 2 conv layers-2 FC layers (input: grayscale, 2D matrix)
    - M14.py: DNN: 4 conv layers- 3 FC layers (input: grayscale, 2D matrix)
--------------------------------------------------------------------------
Notice: Please install all the necessary libraries in your environment for running the code for unet model training: 
In this paper, we use the following Python packages:
- Tensorflow 1.14.0.
- Cuda 10.0 
- kimage 0.15.0.
- scikit-learn 0.21.3.
- keras  2.3.0.
- numpy 1.16.5.
