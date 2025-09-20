
# Real Time Emotion Recognition (mini-Xception)

A Pytorch implementation of "Real-time Convolutional Neural Networks for Emotion and Gender Classification" (mini-Xception)  



#### How to Install
```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install torch torchvision opencv-python numpy
```
**Note:** The code works with various PyTorch versions. If you encounter version conflicts, you can install compatible versions.

#### install opencv & dnn from source (optional)
Both opencv dnn & haar cascade are used for face detection, if you want to use haar cascade you can skip this part.

install dependencies 
```
$ sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
$ sudo apt-get install libv4l-dev libxvidcore-dev libx264-dev
$ sudo apt-get install libgtk-3-dev
$ sudo apt-get install libatlas-base-dev gfortran
```
Download & install opencv with contrib modules from source
```
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip
unzip opencv.zip
unzip opencv_contrib-4.2.0.zip
mkdir -p build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.2.0/modules ../opencv
cmake --build .
```
if you have any problems, refere to [Install opencv with dnn from source](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)

if you **don't** want to use **dnn** modules just setup opencv with regular way
```
sudo apt-get install python3-opencv
```

## Quick Start - Real-Time Emotion Detection

### 🚀 **Easiest Way to Run (Recommended)**
```bash
# Simple emotion output
python emotion_stream.py --format simple

# JSON output for backend integration
python emotion_stream.py --format json

# With video window (for debugging)
python emotion_stream.py --format json --show-video
```

### 📊 **Output Examples**
**Simple Format:**
```
Emotion: Happy (Confidence: 0.849)
Emotion: Neutral (Confidence: 0.756)
```

**JSON Format:**
```json
{
  "emotion": "Happy",
  "confidence": 0.849,
  "probabilities": {
    "Angry": 0.05,
    "Disgust": 0.02,
    "Fear": 0.01,
    "Happy": 0.85,
    "Sad": 0.03,
    "Surprise": 0.02,
    "Neutral": 0.02
  },
  "face_detected": true,
  "timestamp": 1704112200.123
}
```

### 🎯 **Command Options**
```bash
# Use Haar cascade detector (alternative to DNN)
python emotion_stream.py --format json --haar

# Use custom model weights
python emotion_stream.py --format json --pretrained path/to/weights.pth.tar

# Show help
python emotion_stream.py --help
```

---

## Visual Testing Demo
##### Live camera demo with video window (for visual testing)
```bash
# Original demo with video window
python camera_demo.py

# Use Haar cascade detector instead of DNN
python camera_demo.py --haar

# Test on image
python camera_demo.py --image --path path/to/image.jpg

# Test on video
python camera_demo.py --path path/to/video.mp4
```

### Test 
##### image test
```
# replace $PATH_TO_IMAGE with your relative(or global) path to the image 
$ python3 camera_demo.py --image --path PATH_TO_IMAGE
```
##### video test
```
$ python3 camera_demo.py --path PATH_TO_VIDEO
```


#### Face Preprocessing
- Histogram Equalization for iliumination normalization 
- Face Alignment using dlib landmarks
##### Demo

![2](https://user-images.githubusercontent.com/35613645/116496346-22b09480-a8a5-11eb-9715-cefb41d221cc.gif)


### FER2013 Dataset
The data consists of **48x48 pixel grayscale** images of faces. and their emotion shown in the facial expression in to one of seven categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral), The training set consists of **28,709 examples**. The public test set consists of **3,589 examples**.

[Download FER2013](https://www.kaggle.com/deadskull7/fer2013)
- create a folder called "data/" in project root
- put the "fer2013.csv" in it


#### Visualize dataset
Visualize dataset examples with annotated landmarks & head pose 
```cmd
# add '--mode' option to determine the dataset to visualize
$ python3 visualization.py
```
#### Tensorboard 
Take a wide look on dataset examples using tensorboard
```
$ python3 visualization.py --tensorboard
$ tensorboard --logdir checkpoint/tensorboard
```
![Screenshot 2021-04-01 20:05:42](https://user-images.githubusercontent.com/35613645/113335766-aff3de00-9325-11eb-8c07-66379e53a65d.png)



#### Testing
```
$ python3 test.py
```

#### Training 
```
$ python3 train.py
```
#### Evaluation
```
$ python3 train.py --evaluate
```
will show the confision matrix

![Screenshot 2021-04-01 20:13:14](https://user-images.githubusercontent.com/35613645/113336651-04e42400-9327-11eb-8aa1-d52d78eb0ad5.png)

#### Folder structure    
    ├── emotion_stream.py		# 🆕 Real-time emotion detection stream (MVP)
    ├── camera_demo.py			# Visual testing demo with video window
    ├── model					# model's implementation
    ├── data					# data folder contains FER2013 dataset
    ├── train					# train on FER2013 dataset 
    ├── test					# test on 1 example
    ├── face_detector			# contain the code of face detection (dnn & haar-cascade)
    ├── face_alignment			# contain the code of face alignment using dlib landmarks


#### Refrences
Deep Learning on Facial Expressions Survey
- https://arxiv.org/pdf/1804.08348.pdf

ilimunation normalization (histogram / GCN / Local Norm)
- https://www.sciencedirect.com/science/article/pii/S1877050917320860

Tensorflow Implementation
- https://github.com/oarriaga/face_classification/tree/master

Inception (has some used blocks)
- https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

Xception
- https://towardsdatascience.com/review-xception-with-depthwise-separable-convolution-better-than-inception-v3-image-dc967dd42568

Pytorch GlobalAvgPooling
- https://paperswithcode.com/method/global-average-pooling



