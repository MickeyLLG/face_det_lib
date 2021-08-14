from Interface import face_detector
import onnxruntime as ort
import cv2
import numpy as np


class pig_face_detector(face_detector):
    def __init__(self):
        self.session = ort.InferenceSession("./det_pig/models/detector.onnx", None)
        self.x0 = self.session.get_inputs()[0].name
        self.x1 = self.session.get_inputs()[1].name
        self.output = [self.session.get_outputs()[i].name for i in range(3)]
        self.input_size = (512, 512)
        self.thresh = 0.5

    def det_faces(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, scale = pig_face_detector.preprocess(img, self.input_size[0], self.input_size[1])
        img = np.expand_dims(img, 0).astype(np.float32)

        boxes, scores, num_boxes = self.session.run(self.output, {self.x0: img, self.x1: np.array(False)})
        num_boxes = num_boxes[0]
        boxes = boxes[0][:num_boxes]
        scores = scores[0][:num_boxes]
        to_keep = scores > self.thresh
        boxes = boxes[to_keep]
        boxes = boxes * self.input_size[0] / scale

        boxes_return = [[boxes[i][1], boxes[i][0], boxes[i][3] - boxes[i][1], boxes[i][2] - boxes[i][0]]
                        for i in range(len(boxes))]
        return boxes_return

    @staticmethod
    def preprocess(image, target_height, target_width):
        h, w, c = image.shape

        bimage = np.zeros(shape=[target_height, target_width, c], dtype=image.dtype) + np.array([123., 116., 103.],
                                                                                                dtype=image.dtype)
        long_side = max(h, w)

        scale = target_height / long_side

        image = cv2.resize(image, None, fx=scale, fy=scale)

        h_, w_, _ = image.shape
        bimage[:h_, :w_, :] = image

        return bimage, scale


if __name__ == '__main__':
    d = pig_face_detector()
