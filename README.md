
# Project Title

### Vehicle 👁️


## Group Members

| Name | SapID | Email |
| ---------|----------|----------|
|  Sanayna Vohra| 021-20-52856| Sanayna.52856@iqra.edu.pk|
| Saadullah Khan| 021-21-55391| Saad.55391@iqra.edu.pk|
| Sultan Abdullah Siddiqui| 021-21-55170| Sultan.55170@iqra.edu.pk|



## Project Overview

Welcome to the Vehicle Detection, Number Plate Recognition, and Vehicle Counting (VDNVR) system, this project presents a cutting-edge solution powered by Computer Vision and Deep Learning techniques. 

The first component of the project is vehicle detection. This involves developing an algorithm that can identify and locate vehicles within the images. The algorithm will use object detection techniques, advanced methods Deep Learning-based approaches i.e., You Only Look Once (YOLO). The system will detect vehicles regardless of their size, shape, or orientation, enabling it to handle different types of vehicles. In this project we are working on six types of classes bus, car, motorbike, truck, auto, chingchi.

Once the vehicles are detected, the next step is to count them accurately. The system will employ object tracking techniques to maintain a continuous count of vehicles as they enter or exit the monitored area. Various tracking algorithms like centroid tracking will be used to track vehicles over time. The counting component will provide real-time updates of the total number of vehicles in the scene.

In addition to vehicle detection and counting, the system will also extract number plate information from the detected vehicles. This will involve developing an algorithm that can accurately identify and localize license plates within the video frames. The algorithm use techniques like Character Segmentation, Optical Character Recognition (OCR), Deep Learning (DL)-based methods to recognize the characters on the license plates. The extracted information can be used for further analysis or record-keeping purposes.

The project will involve implementing the algorithms for vehicle detection, counting, and number plate detection using a programming language Python and utilizing relevant Computer Vision (CV) libraries such as OpenCV and TensorFlow.

The performance of the system will be evaluated based on various metrics, including detection accuracy, counting accuracy, and license plate recognition accuracy. The system will be tested using diverse datasets with different lighting conditions, weather conditions, and traffic densities. The evaluation will help optimize the algorithms and fine-tune the system to achieve reliable and accurate results in real-world scenarios.

## Features

1. #### Vehicle Detection
The system incorporates state-of-the-art object detection models i.e, YOLO (You Only Look Once), to accurately detect and locate vehicles within the input image.

2. #### Number Plate Recognition
For law enforcement, parking management, and security applications, the system includes a sophisticated number plate recognition component. It utilizes Optical Character Recognition (OCR) techniques and deep learning models to read and interpret number plates, enabling automated identification and tracking of vehicles.

3. #### Vehicle Counting
Accurate vehicle counting is essential for traffic analysis, congestion management, and parking space availability assessment. Our system employs advanced algorithms to count vehicles as they enter and exit a designated area, providing real-time and historical data for decision-making.
## Data Collection
We have amassed a diverse dataset consisting of 876  images featuring various types of vehicles. Additionally, our collection comprises 850 high-quality images of license plates. Most of this extensive dataset was acquired through daily captures of vehicles in real-world settings, as well as from external resources to ensure its richness and diversity. This comprehensive dataset serves as a valuable resource for a wide range of applications, from Computer Vision and Deep Learning.



## Data Annotation

We have selected Roboflow online tool as our preferred platform for annotating and labeling the data we've gathered. It has proven to be an invaluable tool for our data annotation and labeling needs, and we selected it primarily for its user-friendly environment. Here are some key benefits of using RoboFlow for our projects:

1. #### Enhanced Accuracy:
We've created two separate projects within RoboFlow, one for vehicle detection and another for license plate recognition, to leverage its exceptional accuracy. RoboFlow's precision and robust annotation tools have contributed significantly to the overall quality of our labeled data.

2. #### Time Efficiency: 
With RoboFlow's automation and efficient workflows, we have been able to save time during the data annotation process. This time-saving feature allows us to focus on other critical aspects of our projects.

3. #### Collaboration: 
RoboFlow offers collaborative features that facilitate teamwork and enable multiple contributors to work on the same projects simultaneously. This collaboration capability enhances our overall productivity and ensures consistent results.

4. #### Data Management: 
RoboFlow simplifies the organization and management of our labeled data, making it easy to maintain and update our datasets as needed. This capability is crucial for long-term project success.

### Vehicle

![Vehicle](https://github.com/sanaynavohra/Vehicle_Eye/blob/master/outputs/annoveh.png?raw=true)



### Num_plate


![num_plate](https://github.com/sanaynavohra/Vehicle_Eye/blob/master/outputs/annoplate.png?raw=true)




## Image Dataset

### Vehicle_Detection 

![Vehicle_Detection](https://github.com/sanaynavohra/Vehicle_Eye/blob/master/outputs/vehh.png?raw=true)

### License_plate


![num_plate](https://github.com/sanaynavohra/Vehicle_Eye/blob/master/outputs/noplate.png?raw=true)

## Roboflow Custom Dataset

1. #### vehicle_detection
https://universe.roboflow.com/customdataset-0fkml/vehicle_detection-uey8e

2. #### license_plate
https://universe.roboflow.com/customdataset-0fkml/license_plate-ahokr
## Model Training

### Vehicle Detection

#### Table of Contents
- Setup
- Data Preparation
- Model Training
- Validation
- Inference
 ---
#### Setup

Setting up the environment, dependencies, and prerequisites.

```python
from google.colab import drive
drive.mount('/content/drive')
```
```python
import os
absolute_path = '/content/drive/MyDrive/vehicle_detection_training'
os.chdir(absolute_path)

!pwd
```
```python 
!pip install ultralytics
from ultralytics import YOLO
```
---
#### Data Preparation

Upload Processed Data to Colab   
Use the following command to download the processed data from Roboflow and upload it to your Colab environment:

```python
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="AXDp1w3S4RiBVG5Ly8bv")
project = rf.workspace("customdataset-0fkml").project("vehicle_detection-uey8e")
dataset = project.version(2).download("tfrecord")
```
---
 #### Model Training
 ### Vehicle detection


 Run Training Script:
  ```python
  !yolo task=detect mode=train model=yolov8m.pt data={dataset.location}/data.yaml epochs=100 imgsz=640
  ```
Monitor Training:

![Training](https://github.com/sanaynavohra/Vehicle-Eye/blob/main/New%20folder/training_veh.jpg?raw=true)

![Training](https://github.com/sanaynavohra/Vehicle_Eye/blob/master/outputs/Picture2.png?raw=true)





 #### Validation
 Run Validation Script:
 ```pyhton
 !yolo task=detect mode=val model= /content/drive/MyDrive/vehicle_detection_training/runs/detect/train/weights/best.pt  data={dataset.location}/data.yaml
  
 ```

 ![Training](https://github.com/sanaynavohra/Vehicle-Eye/blob/main/New%20folder/image4.png?raw=true)

 #### Inference
 Run Inference Script:
```python
!yolo task=detect mode=predict model= /content/drive/MyDrive/vehicle_detection_training/runs/detect/train/weights/best.pt conf=0.25 source='/content/drive/MyDrive/vehicle_detection_training/data/bike (17).jpg'
```
Result
![Training](https://github.com/sanaynavohra/Vehicle-Eye/blob/main/New%20folder/image9.jpeg?raw=true)

---

### license Plate

#### Table of Contents
- Setup
- Data Preparation
- Model Training
- Validation
- Inference
 ---
#### Setup

Setting up the environment, dependencies, and prerequisites.

```python
from google.colab import drive
drive.mount('/content/drive')
```
```python
import os
absolute_path = '/content/drive/MyDrive/vehicle_detection_training'
os.chdir(absolute_path)

!pwd
```
```python 
!pip install ultralytics
from ultralytics import YOLO
```
---
#### Data Preparation

Upload Processed Data to Colab   
Use the following command to download the processed data from Roboflow and upload it to your Colab environment:

```python
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="AXDp1w3S4RiBVG5Ly8bv")
project = rf.workspace("customdataset-0fkml").project("license_plate-ahokr")
dataset = project.version(6).download("yolov8")
```
---
 #### Model Training

 Run Training Script:
  ```python
 !yolo task=detect mode=train model=yolov8m.pt data={dataset.location}/data.yaml epochs=150 imgsz=640
  ```
Monitor Training:

![Training](https://github.com/sanaynavohra/Vehicle-Eye/blob/main/New%20folder/num_plate/1.jpg?raw=true)

![Training](https://github.com/sanaynavohra/Vehicle_Eye/blob/master/outputs/confusionmat.png?raw=true)


 #### Validation
 Run Validation Script:
 ```pyhton
!yolo task=detect mode=val model=  data={dataset.location}/data.yaml/content/drive/MyDrive/NO_PLATE/runs/detect/train/weights/best.pt
 ```
 ![Training](https://github.com/sanaynavohra/Vehicle_Eye/blob/master/outputs/licenseplate.jpeg?raw=true)
 
 #### Inference
 Run Inference Script:
```python
!yolo task=detect mode=predict model=/content/drive/MyDrive/NO_PLATE/runs/detect/train/weights/best.pt  conf=0.25 source='/content/drive/MyDrive/NO_PLATE/IMG_4949.JPG'
```
Result
![Training](https://github.com/sanaynavohra/Vehicle-Eye/blob/main/New%20folder/num_plate/7.jpeg?raw=true)











