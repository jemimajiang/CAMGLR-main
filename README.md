# CAMGLR-main

## Requirements
### Dependencies
* Python 3.5+
* Chainer 5.0.0+
* ChainerRL 0.5.0+
* Cupy 5.0.0+
* OpenCV 4.2.0.32
* Numpy 1.19.5
* Scipy 1.5.4

### It was ran and tested under the following OSs:
* Ubuntu 16.04 with NVIDIA TeslaT4 GPU

## Preparing Data
1. To build **training** dataset, you'll also need following datasets. All the images needs to be converted to **gray scale**.
* [Middlebury2006](https://vision.middlebury.edu/stereo/data/scenes2006/)
* [BSD68](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) and converted the images to gray scale by using the cvtColor function in OpenCV.
* [Waterloo](https://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)

2. To build **validation/testing** dataset, you'll also need following datasets. All the images needs to be cropped into a square, and resize to **70*70**.
* [Middlebury2005](https://vision.middlebury.edu/stereo/data/scenes2005/)
* [MNIST](https://opendatalab.com/OpenDataLab/MNIST/tree/main)

3.Structure of the datasets should be：
  ```
  ├── datasets
      ├──BSD68       
      ├──Middlebury        
      ├──MNIST
      ├──pristine_images
  ```

## Getting Started:
### Usage
* Training
    * To train this code.
    ```
    python train.py
    ```

    * To train with different settings, modify ```LEARNING_RATE```, ```GAMMA```, ```GPU_ID```, ```N_EPISODES```, ```CROP_SIZE```, ```TRAIN_BATCH_SIZE```, ```EPISODE_LEN``` as you need.

* Testing
    * To test this code.
    ```
    python test.py
    ```
    * To test with different settings, modify ```GPU_ID```, ```CROP_SIZE```, ```TRAIN_BATCH_SIZE```, ```EPISODE_LEN``` as you need.
      
## References
We used the publicly avaliable models of [[Zhang+, CVPR17]](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_Learning_Deep_CNN_CVPR_2017_paper.html) as the initial weights of our model except for the convGRU and the last layers. We obtained the weights from [here](https://github.com/cszn/DnCNN) and converted them from MatConvNet format to caffemodel in order to read them with Chainer.
