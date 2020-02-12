import onnxruntime as ort
import cv2
import numpy as np
from Interface import landmark_detector

session = ort.InferenceSession("det_pfld/graphs/pfld-lite.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


class pfld_landmark_detector(landmark_detector):
    def det_landmarks(self, img, faces):
        landmarks=[]
        for face in faces:
            face_img=img[face[1]:face[1]+face[3], face[0]:face[0]+face[2], :]
            pred=self.det_landmarks_per_face(face_img)
            pred[:,0]+=face[0]
            pred[:,1]+=face[1]
            marks=[]
            for l in range(pred.shape[0]):
                marks.append([pred[l,0],pred[l,1]])
            landmarks.append(marks)
        return landmarks
    def det_landmarks_per_face(self, face_img):
        h, w, d = face_img.shape
        image = cv2.resize(face_img, (96, 96))
        data = np.array(image, np.float32)
        data = (data - 123.0) / 58.0
        data = np.transpose(data, (2, 0, 1))
        data = np.reshape(data, (1, 3, 96, 96))
        pred_onnx = session.run([output_name], {input_name: data})[0]
        pred = np.reshape(pred_onnx, (-1, 2))
        pred[:, 0] *= w / 96
        pred[:, 1] *= h / 96
        return pred
