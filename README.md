
 Face recognition online service and do age/gender detection in TensorFlow
==========================================================

## Goal

Face recognition online service, allow user realtime training it and do Age/Gender detection in Tensorflow.

### Currently Supported Models

- Face detection and recognition using Dlib(Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems. [https://github.com/davisking/dlib](https://github.com/davisking/dlib))
- Face detection and recognition using MTCNN&Facenet(comming)
- Gil Levi and Tal Hassner, Age and Gender Classification Using Convolutional Neural Networks, IEEE Workshop on Analysis and Modeling of Faces and Gestures (AMFG), at the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, June 2015_
    - http://www.openu.ac.il/home/hassner/projects/cnn_agegender/
    - https://github.com/GilLevi/AgeGenderDeepLearning
- Inception v3 with fine-tuning
    - This will start with an inception v3 checkpoint, and fine-tune for either age or gender detection 

### Running

Here is a run using dlib for face-detector & recognition and  Age classification on the latest checkpoint in a directory using 12-look (all corners + center + resized, along with flipped versions) averaging:

```
$ python app.py 
```

#### Face Detection

If you have an image with one or more frontal faces, you can run a face-detector upfront, and each detected face will be chipped out and run through classification individually.  A variety of face detectors are supported including OpenCV, dlib and MTCNN

To use dlib, you will need to install it and grab down the model:

```
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2(dlib_face_recognition_resnet_model_v1.dat)
bunzip2 bunzip2 shape_predictor_68_face_landmarks.dat.bz2(put into models/)
pip install dlib
```

#### Prediction with fine-tuned inception model

gender_age.py  with an inception fine-tuned model is for Age and gender classification(put into models/)

### Pre-trained Checkpoints
You can find a pre-trained age checkpoint for inception here:

https://drive.google.com/drive/folders/0B8N1oYmGLVGWbDZ4Y21GLWxtV1E

A pre-trained gender checkpoint for inception is available here:

https://drive.google.com/drive/folders/0B8N1oYmGLVGWemZQd3JMOEZvdGs


