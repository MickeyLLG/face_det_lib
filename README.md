# face_det_lib
## Capabilities
-  Use models provided by biubug, centerface, dlib, linzaer, mobileface, mtcnn, zqcnn, pig to detect faces.
-  Use dilb, pfld, pig, zqcnn models or the two L106Net models to detect key points of the face (ranging from 68-106 points)
-  Support the detection of pictures, videos and camera input
-  Support the visualization and export of test results
-  Support recording inference time
## Environment
-  windows10
-  python3.6
-  numpy==1.20.2
-  opencv-python==4.5.1.48
-  dlib==19.6.1（dlib）
-  mtcnn==0.1.0（mtcnn）
-  pytorch==1.4.0（biubug）
-  onnx==1.9.0（linzaer）
-  onnxruntime==1.8.1（linzaer, pfld, centerface, frda, pig）
-  mxnet==1.5.0（mobileface, zqcnn）
-  tensorflow==1.5.0（cnn）
## Installation
The project contains all pre-trained models that may be used, just clone them directly
```bash
# From your favorite development directory:
git clone https://github.com/MickeyLLG/face_det_lib.git
```
## Run
Video files, camera serial numbers or image files need to be fed in as parameters. If nothing is fed in, it will detect test/test_15fps.avi as default
### Videos input

Any video format supported by OpenCV is available (`mp4`, `avi` etc.):

```bash
python demo.py --video /path/to/video.mp4 --save_path /path/to/save.mp4 --face_det fd --landmark_det ld
```
### Camera input

Need to declare the serial number of the camera used: 

```bash
python demo.py --cam 0 --save_path /path/to/save.mp4 --face_det fd --landmark_det ld
```
### Image input

Any image format supported by OpenCV is available (`jpg`, `jpeg` etc.):

```bash
python demo.py --image /path/to/image.jpg --save_path /path/to/save.jpg --face_det fd --landmark_det ld
```
### Face detection method selection
`--face_det`parameter determines the method used for face detection, and the optional methods include(`pig` is recommended for better performance)：  
`dlib`,`mtcnn`,`linzaer`,`centerface`,`biubug`,`mobileface`,`zqmtcnn`,`pig`
### Selection of face key points detection method
`--landmark_det` parameter determines the method used for face key point detection. The optional methods include(`pig` is recommended for better performance):
`dlib`,`pfld`,`L106Net112`,`L106Net96`,`cnn`,`frda`,`pig`
## Reference
- [**pfld**](https://github.com/xindongzhang/MNN-APPLICATIONS "pfld")
- [**linzaer**](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB "linzaer")
- [**mobileface**](https://github.com/becauseofAI/MobileFace "mobileface")
- [**biubug**](https://github.com/biubug6/Face-Detector-1MB-with-landmark "biugbug")
- [**zqcnn**](https://github.com/zuoqing1988/train-mtcnn-head "zqcnn")
- [**centerface**](https://github.com/Star-Clouds/centerface "centerface")
- [**cnn**](https://github.com/yinguobing/head-pose-estimation "cnn")
- [**frda**](https://github.com/Star-Clouds/FRDA "frda")
- [**pig**](https://github.com/610265158/Peppa_Pig_Face_Engine "pig")

***
### Chinese

## 实现功能

-  使用biubug、centerface、dlib、linzaer、mobileface、mtcnn、zqcnn提供的模型来检测人脸
-  使用dilb、pfld模型和zqcnn提供的两种L106Net模型来检测面部关键点（68-106点不等）
-  支持对图片、视频以及摄像头输入的检测
-  支持检测结果可视化并导出
-  支持记录推理时间
## 测试过的运行环境
-  windows10
-  python3.6
-  numpy==1.20.2
-  opencv-python==4.5.1.48
-  dlib==19.6.1（dlib）
-  mtcnn==0.1.0（mtcnn）
-  pytorch==1.4.0（biubug）
-  onnx==1.9.0（linzaer）
-  onnxruntime==1.8.1（linzaer, pfld, centerface, frda, pig）
-  mxnet==1.5.0（mobileface, zqcnn）
-  tensorflow==1.5.0（cnn）
## 安装
项目内包含所有可能用到的预训练模型，直接克隆即可
```bash
# From your favorite development directory:
git clone https://github.com/MickeyLLG/face_det_lib.git
```
## 运行
视频文件、摄像头序号或图像文件需要作为参数传入。如果没有传入，则会默认检测test/test_15fps.avi
### 视频输入

OpenCV支持的任何视频格式均可 (`mp4`, `avi` etc.):

```bash
python demo.py --video /path/to/video.mp4 --save_path /path/to/save.mp4 --face_det fd --landmark_det ld
```
### 摄像头输入

需要声明使用的摄像头序号:

```bash
python demo.py --cam 0 --save_path /path/to/save.mp4 --face_det fd --landmark_det ld
```
### 图片输入

OpenCV支持的任何图片格式均可 (`jpg`, `jpeg` etc.)，路径中不要有中文:

```bash
python demo.py --image /path/to/image.jpg --save_path /path/to/save.jpg --face_det fd --landmark_det ld
```
### 人脸检测方法选择
`--face_det`参数决定了人脸检测使用的方法，可选方法包括(推荐`pig`，效果较好)：  
`dlib`,`mtcnn`,`linzaer`,`centerface`,`biubug`,`mobileface`,`zqmtcnn`,`pig`

### 人脸关键点检测方法选择
`--landmark_det`参数决定了人脸关键点检测使用的方法，可选方法包括(推荐`pig`，效果较好)：  
`dlib`,`pfld`,`L106Net112`,`L106Net96`,`cnn`,`frda`,`pig`

## 参考
- [**pfld**](https://github.com/xindongzhang/MNN-APPLICATIONS "pfld")
- [**linzaer**](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB "linzaer")
- [**mobileface**](https://github.com/becauseofAI/MobileFace "mobileface")
- [**biubug**](https://github.com/biubug6/Face-Detector-1MB-with-landmark "biugbug")
- [**zqcnn**](https://github.com/zuoqing1988/train-mtcnn-head "zqcnn")
- [**centerface**](https://github.com/Star-Clouds/centerface "centerface")
- [**cnn**](https://github.com/yinguobing/head-pose-estimation "cnn")
- [**frda**](https://github.com/Star-Clouds/FRDA "frda")
- [**pig**](https://github.com/610265158/Peppa_Pig_Face_Engine "pig")

