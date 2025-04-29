# SDAS Prediction in Aluminum Casting using Computer Vision

## Introduction

This repository is the result of my internship at GFaI (Gesellschaft für Förderung angewandter Informatik e.V.).  
The goal is to compare different approaches to predict Secondary Dendrite Arm Spacing (SDAS) in aluminum casting using Computer Vision techniques.

The project includes two distinct approaches:

1. CNN-based Regression  
   Inspired by the paper "Casting Microstructure Inspection Using Computer Vision" (https://www.mdpi.com/2075-4701/11/5/756), this approach uses a Convolutional Neural Network (CNN) to predict a scaled SDAS value from grayscale 200x200 pixel images.  
   I re-implemented the CNN architecture described in the paper and trained it using a dataset of 204 labeled images provided by BMW.

2. Hybrid Detection-Based Approach  
   This method combines segmentation and clustering:
   - It uses the CellPose model to detect the boundaries of dendritic cells.
   - A clustering algorithm is then applied to detect individual dendrites and calculate the average SDAS for each one.

Both methods are trained and evaluated within this repository.

## How to Run

> Make sure you have Anaconda installed.

### CNN Approach

1. Open Anaconda Prompt  
2. Navigate to the `environments/` folder of this repository  
3. Create the environment:  
   conda env create -f tfgpu.yml -n your_env_name  
4. Activate the environment:  
   conda activate your_env_name  
5. Start Jupyter Notebook:  
   jupyter notebook  
6. Open the folder: `notebooks_CNN`  
7. Open and run: `CNN_SDAS_Prediction_Pipeline.ipynb`  

### Hybrid Approach

1. Open Anaconda Prompt  
2. Navigate to the `environments/` folder  
3. Create the environment:  
   conda env create -f hybrid_env.yml -n your_env_name  
4. Activate the environment:  
   conda activate your_env_name  
5. Start Jupyter Notebook:  
   jupyter notebook  
6. Open the folder: `notebooks_HYBRID`  
7. Open and run: `HYBRID_SDAS_Prediction_Pipeline.ipynb`  

## Acknowledgments

- BMW – for providing the dataset of annotated casting microstructure images.  
- GFaI – for supporting this project during my internship.  
- CellPose – for the segmentation model used in the hybrid approach.  
- MDPI Paper – for the CNN-based methodology: https://www.mdpi.com/2075-4701/11/5/756  

## License

MIT, Apache 2.0

## Contact

For questions or contributions, feel free to open an issue or contact me at j.kleinau{at}proton.me .
