import sys
import cv2

sys.path.append('..')
from Interface import face_detector
sys.path.append('det_centerface')
from centerface import CenterFace

size = (180, 180)  # (w,h) 传入网络的图像大小，可以修改，传入的越大越准，但速度越慢，反之亦然
centerface = CenterFace(height=size[1], width=size[0])


class centerface_face_detector(face_detector):
    def det_faces(self, img):
        h, w=img.shape[:2]
        w_scale=w/size[0]
        h_scale=h/size[1]
        frame = cv2.resize(img, size)
        dets, lms = centerface(frame, threshold=0.5)
        faces=[]
        for det in dets:
            boxes, score = det[:4], det[4]
            x,y,r,b=int(boxes[0]*w_scale), int(boxes[1]*h_scale),int(boxes[2]*w_scale), int(boxes[3]*h_scale)
            b = int(b + (b - y) * 0.05)
            faces.append([x, y, r-x, b-y])
        return faces

