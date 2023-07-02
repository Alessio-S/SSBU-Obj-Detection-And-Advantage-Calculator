# SSBU-Obj-Detection-And-Advantage-Calculator
**Custom Object Detection** for Super Smash Bros. Ultimate (using TensorFlow 2) & positional **Advantage Calculator**.

## About
I have trained a Deep Learning AI, using [Tensorflow's 2.0 Custom Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md), to identify and locate Super Smash Bros. Ultimate Characters from a video source, and use the data to calculate the relative positional advantage (in %) of one character to the other(s).

The model I trained lacks a large amount of data to work properly, so I used [Matthew Tafazoli's SSBM-Custom-Object-Detection](https://github.com/MatthewTafazoli/SSBM-Custom-Object-Detection) model, well trained to identify characters in Super Smash Bros. Melee, as a base to try the positional advantage calculator. Here is an output example:
![Peach-Fox Advantage Calulator](/assets/Peach-Fox_adv_ex.jpg)

To train the model, I run a script on an MP4 file that converts it to a list of images. I then annotate every image using [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/) which I then export as a single .json file. Using another script, I convert the .json file in a .tfrec file to train the model with, in the [my_model/pipeline.config](my_model/pipeline.config) file, required by Tensorflow's 2.0 API.

## Installation
Please refer to the [Official Tensorflow Installation Guide](https://www.tensorflow.org/install) to avoid any compatibility issues.
I am curently running on `Tensorflow 2.12.3` with `Keras 2.12.0` and `Python 3.10.6`. I am using `pip 23.1.2` to install the needed packages.
Here is a full list of packages you may need to install/upgrade using `pip`:
```pip install tensorflow (also installs keras & tensorboard)```
```pip install protobuf```
```pip install opencv-python```
```pip install numpy```
```pip install pandas```
```pip install model-lib```
```pip install grpcio```
```pip install grpcio-tools```

(Optional)
```pip install virtualenv (to isolate your pip packages and depencies)```
```

## Usage
