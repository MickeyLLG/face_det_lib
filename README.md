# face_det_lib
## 实现功能
-  使用biubug、centerface、dlib、linzaer、mobileface、mtcnn、zqcnn提供的模型来检测人脸
-  使用dilb、pfld模型和zqcnn提供的两种L106Net模型来检测面部关键点（68-106点不等）
-  支持对图片、视频以及摄像头输入的检测
-  支持检测结果可视化并导出
-  支持记录推理时间
## 测试过的运行环境
-  windows10
-  python3.6
-  numpy==1.16.6
-  opencv-python==4.1.2
-  dlib==19.6.1（dlib）
-  mtcnn==0.1.0（mtcnn）
-  pytorch==1.4.0（biubug）
-  onnx==1.5.0（linzaer，pfld，centerface，frda）
-  mxnet==1.5.0（mobileface，zqcnn）
-  tensorflow==1.5.0（cnn）.
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
`--face_det`参数决定了人脸检测使用的方法，可选方法包括：  
`dlib`,`mtcnn`,`linzaer`,`centerface`,`biubug`,`mobileface`,`zqmtcnn`
### 人脸关键点检测方法选择
`--landmark_det`参数决定了人脸关键点检测使用的方法，可选方法包括：  
`dlib`,`pfld`,`L106Net112`,`L106Net96`,`cnn`,`frda`
## 参考
- [**pfld**](https://github.com/xindongzhang/MNN-APPLICATIONS "pfld")
- [**linzaer**](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB "linzaer")
- [**mobileface**](https://github.com/becauseofAI/MobileFace "mobileface")
- [**biubug**](https://github.com/biubug6/Face-Detector-1MB-with-landmark "biugbug")
- [**zqcnn**](https://github.com/zuoqing1988/train-mtcnn-head "zqcnn")
- [**centerface**](https://github.com/Star-Clouds/centerface "centerface")
- [**cnn**](https://github.com/yinguobing/head-pose-estimation "cnn")
- [**frda**](https://github.com/Star-Clouds/FRDA "frda")

