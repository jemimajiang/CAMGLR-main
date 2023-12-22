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
* [Middlebury](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)
* [BSD68](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)
* [Waterloo](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)

2. To build **validation/testing** dataset, you'll also need following datasets. All the images needs to be cropped into a square, and resize to **70*70**.
* [Middlebury](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip)
* [MNIST](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip)

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
