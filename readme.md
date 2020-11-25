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
    
        
## Note:
I'm currently recommending the newer RootPainter software for new segmentation projects:

https://www.biorxiv.org/content/10.1101/2020.04.16.044461v2
https://github.com/Abe404/root_painter
