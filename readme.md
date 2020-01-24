Code for the paper "Segmentation of Roots in Soil with U-Net"

Data is available from https://zenodo.org/record/3527713

Trained model is available from https://zenodo.org/record/3484015

## Install dependencies
    > pip install -r requirements.txt


## Compute and print U-Net test set metrics
    > cd ./src
    > python ./unet/test.py


## Compute Frangi test set metrics
    > cd ./src
    > python ./frangi/test.py
