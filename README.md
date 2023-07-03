# SSBU-Obj-Detection-And-Advantage-Calculator
**Custom Object Detection** for Super Smash Bros. Ultimate (using TensorFlow 2) & positional **Advantage Calculator**.

## About
I have trained a Deep Learning AI, using [Tensorflow's 2.0 Custom Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md), to identify and locate Super Smash Bros. Ultimate Characters from a video source, and use the data to calculate the relative positional advantage (in %) of one character to the other(s).

The model I trained lacks a large amount of data to work properly, so I used [Matthew Tafazoli's SSBM-Custom-Object-Detection](https://github.com/MatthewTafazoli/SSBM-Custom-Object-Detection) model, well trained to identify characters in Super Smash Bros. Melee, as a base to try the positional advantage calculator. Here is an output example:

![Peach-Fox Advantage Calulator](/assets/Peach-Fox_adv_ex.jpg)

To train the model, I run a script on an MP4 file that converts it to a list of images. I then annotate every image using [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/) which I then export as a single .json file. Using another script, I convert the .json file in a .tfrecord file to train the model with, in the [my_model/pipeline.config](my_model/pipeline.config) file, required by Tensorflow's 2.0 API.

## Installation
Please refer to the [Official Tensorflow Installation Guide](https://www.tensorflow.org/install) to avoid any compatibility issues.
I am curently running on `Tensorflow 2.12.3` with `Keras 2.12.0` and `Python 3.10.6`. I am using `pip 23.1.2` to install the needed packages.

Here is a full list of packages you may need to install/upgrade using `pip`:
```
pip install tensorflow # Also installs keras & tensorboard
```
```
pip install protobuf
```
```
pip install opencv-python
```
```
pip install numpy
```
```
pip install pandas
```
```
pip install model-lib
```
```
pip install grpcio
```
```
pip install grpcio-tools
```

(Optional)
```
pip install virtualenv # To isolate your pip packages and depencies
```

## Contents
`my_model/pipeline.config` - intializes the parameters for training the model with TF 2

`my_model/saved_model/saved_model.pb` - The exported model after it has been trained on the images set

`annotations/label_map.pbtxt` - Label map file listing your class names paired with a unique index, necessary for *pipeline.config*

`scripts/create_label_map.py` - Script to create the *label_map* file

`scripts/JsonToTFrecord.py` - Takes the '.json' files output from the VGG Image Annotator and turns them into '.tfrecord' files. Data is stored separately for each bounding box, so the .tfrecord files grow quite large based on how many bounding boxes are used per image.

`scripts/eval_model.py` - Run this script on an image folder to test the model on those images and save the resultant annotated images

`scripts/exporter_main_v2.py` - Official code from Tensorflow 2 for exporting the model after completing training

`scripts/model_main_tf2.py` - Official code from Tensorflow 2 for training the model on their Custom Object Detection

`scripts/testtfrec.py` - Tests the content of the '.tfrecord' files you output from

## Usage

#### If you've already added the models directories as permanent environment variables, skip steps 1. to 4.

1. Clone the *Tensorflow models repository* from GitHub
```
git clone https://github.com/tensorflow/models.git
```
2. Navigate to the */models/research* directory
```
cd models/research/
```
3. Compile the *Protobufs*
```
protoc object_detection/protos/*.proto --python_out=.
```
4. Add the */models/research* and */models/research/slim* directories to your *PYTHONPATH* environment variable
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

5. Navigate to the *scripts directory*
```
cd scripts/
```

#### Train the model

1. Get the *.json file* for the annotations
   
2. Convert the *.json file* to a *.tfrecord file*
```
${YOUR_PYTHON_VERSION} JsonToTFrecord.py
  --ImgInputDir dir/of/images/used/for/annotations
  --JsonFile path/to/json/file
  --OutputDir path/where/tfrecord/file/will/be/saved
  --OutName "Name of the .tfrecord file"
  --CategoryName "Name of the category used for annotations"
  --train_val_ratio (0, inf) # Ratio for making the .tfrecord file
      (ex: for 0.5 -> only half of the .json file items will be converted)
```
3. Create your *label_map file* with *create_label_map.py*
```
[...]
label_map = convert_classes(['Lucina', 'Roy'])          # Modify as per the classes you want
output_path = "../annotations/"                         # Place where the label_map file will be saved
with open(output_path + "label_map.pbtxt", "w") as f:   # You can change the name here
    [...]
```
4. Modify the *my_model/pipeline.config* file as per your needs
```
[...]
train_input_reader {
  label_map_path: "path/to/your/label_map_file"
  tf_record_input_reader {
    input_path: "${YOUR_TFRECORD_FILE_NAME}_train.tfrecord"
  }
}
[...]
eval_input_reader {
  label_map_path: "path/to/your/label_map_file"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "${YOUR_TFRECORD_FILE_NAME}_val.tfrecord"
  }
}
```
5. Train the model with the *.tfrecord file*
```
${YOUR_PYTHON_VERSION} model_main_tf2.py
```
6. Repeat steps 2. to 5.

#### Export your trained model
```
${YOUR_PYTHON_VERSION} exporter_main_v2.py
```

#### Evaluate the model
1. *Create the data* you want to test the AI on
```
${YOUR_PYTHON_VERSION} create_test_data.py --MP4Path path/to/your/video --OutputDir dir/for/resultant/images --ImgInterval (0,inf)
```
2. *Test the data and save* the resultant images
```
${YOUR_PYTHON_VERSION} eval_model.py --InputImageDir dir/for/the/input/images --OutImgDir dir/for/annotated/images
```

## Limitations
As of this first version, the AI has only been trained to identify up to two characters, which it does quite poorly, due to a lack of training data.
However, the algorhythm to calculate the positional advantage works really good, as long as the AI finds the characters and not some false positive.

