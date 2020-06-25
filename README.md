# Real Time Object/Face Detection Using YOLO-v3 
This project implements a real time object and face detection using YOLO algorithm. You only look once, or YOLO, is one of the fastet object detection algorithm, suitable for real-time detection. This repository contains code for object and face detector based on [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf) which originaly implemented in [YOLOv3](https://github.com/pjreddie/darknet). The first part of this project focuses on object detection algorithm from scratch in pytorch using pre-trained weights. The second part and third part relate to the object detection and face detection algorithm using opencv library using yolo pre-trained weights.
# Requirements
1. python 3.5
2. Opencv 3.4.2
3. pytorch
4. numpy
# Object Detection Example 
![result2](https://user-images.githubusercontent.com/51369142/85570975-b1bca280-b62b-11ea-889d-8e5a406ce775.jpg)
# Face Detection Example 
![Face1](https://user-images.githubusercontent.com/51369142/85573505-ddd92300-b62d-11ea-8597-ace58f6fb7cc.jpg)
# Running the detectors
### Object detection using pytorch
To detect a single or multiple images, first clone the repository 

`
git clone https://github.com/shayantaherian/Object-detection/Pytorch_ObjectDetection/.git
`

Then move to the directory


`
cd Pytorch_ObjectDetection
`

To run image detection

`
python detect.py
`

To run video and real-time webcame

`
python detect_video.py
`

Note that YOLO weights can be downloaded from [yolov3.weights](https://pjreddie.com/darknet/yolo/)

### Object detection using opencv
OpenCV `dnn ` module supports different pre-trained deep learning models and recently YOLO/Darknet has been added to opencv dnn module. To run the detector, first clone to the repository

`
git clone https://github.com/shayantaherian/Object-detection/OpenCV_ObjectDetection/.git
`

Then move to the directory

`
cd OpenCV_ObjectDetection
`

To run image detection

`
python yolo_Opencv.py
`

To run video and real-time webcame

`
python yolo_Opencv_video.py
`
### Face detection using opencv
To run the face detector, first clone to the repository

`
git clone https://github.com/shayantaherian/Object-detection/OpenCV_FaceDetection/.git
`
Then move to the directory 

`
cd OpenCV_FacetDetection
`

Run the image detector

`
python yolo_Opencv_Face.py
`

Run the video/webcame detector

`
python yolo_Opencv_Face_video.py
`

Note that to use yolo for face detection it is required to download yolo face detection weight from [yolov3.weights](https://drive.google.com/file/d/1xYasjU52whXMLT5MtF7RCPQkV66993oR/view). An example of face detection on video:


![ezgif com-video-to-gif](https://user-images.githubusercontent.com/51369142/85692530-62c74980-b6cd-11ea-8a88-1f068b5551d1.gif)

### References
1. [Paperspace: YOLO object detector in PyTorch](https://blog.paperspace.com/tag/series-yolo/)
2. [YOLO object detection using Opencv with Python](https://www.youtube.com/watch?v=h56M5iUVgGs)
