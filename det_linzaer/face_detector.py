import sys
import onnx

sys.path.append('..')
from Interface import face_detector
import cv2
import onnxruntime as ort
import numpy as np

sys.path.append('det_linzaer')
import vision.utils.box_utils_numpy as box_utils

label_path = "det_linzaer/models/voc-model-labels.txt"

onnx_path = "det_linzaer/models/onnx/version-RFB-320.onnx"
class_names = [name.strip() for name in open(label_path).readlines()]

predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
onnx.helper.printable_graph(predictor.graph)

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name
threshold = 0.98


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


class linzaer_face_detector(face_detector):
    def det_faces(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        confidences, boxes = ort_session.run(None, {input_name: image})
        boxes, labels, probs = predict(img.shape[1], img.shape[0], confidences, boxes, threshold)
        faces = []
        for i in range(boxes.shape[0]):
            x, y, r, b = boxes[i, :]
            faces.append([x, y, r - x, b - y])
        return faces
