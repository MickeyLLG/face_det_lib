import mxnet as mx
import sys
import cv2
import numpy as np

sys.path.append('det_zqcnn')
from tools.load_model import load_param
from core.symbol import L106_Net_112,L106_Net96_v2
from Interface import landmark_detector

pretrained_112 = 'det_zqcnn/model/lnet106_112'
epoch_112 = 4070
data_size_112 = 112
pretrained_96 = 'det_zqcnn/model/L106_96_v2s'
epoch_96 = 8000
data_size_96 = 96
ctx = mx.cpu()  # 此处可改为GPU


class L106Net112_landmark_detector(landmark_detector):
    def __init__(self):
        self.sym = L106_Net_112('test')
        self.pretrained = pretrained_112
        self.epoch = epoch_112
        self.data_size = data_size_112
        self.args, self.auxs = load_param(self.pretrained, self.epoch, convert=True, ctx=ctx)
        self.data_shapes = {'data': (1, 3, self.data_size, self.data_size)}
        # mx.cpu(), x=(5,4),grad_req='null'
        self.executor = self.sym.simple_bind(ctx, grad_req='null', **dict(self.data_shapes))

    def det_landmarks(self, img, faces):
        landmarks = []
        for face in faces:
            image = img[face[1]:face[1]+face[3], face[0]:face[0]+face[2], :]
            h, w, d = image.shape
            image = cv2.resize(image, (self.data_size, self.data_size))
            newimg1 = np.zeros(self.data_shapes['data'], dtype=np.float)
            newimg1[0, 0, :, :] = (image[:, :, 0] - 127.5) * 0.0078125
            newimg1[0, 1, :, :] = (image[:, :, 1] - 127.5) * 0.0078125
            newimg1[0, 2, :, :] = (image[:, :, 2] - 127.5) * 0.0078125
            self.args['data'] = mx.nd.array(newimg1, ctx)
            self.executor.copy_params_from(self.args, self.auxs)
            out_list = [[] for _ in range(len(self.executor.outputs))]
            self.executor.forward(is_train=False)
            for o_list, o_nd in zip(out_list, self.executor.outputs):
                o_list.append(o_nd.asnumpy())
            out = list()
            for o in out_list:
                out.append(np.vstack(o))
            marks = out[0]
            for j in range(int(len(marks) / 2)):
                if marks[2 * j] > 1:
                    marks[2 * j] = 1
                if marks[2 * j] < 0:
                    marks[2 * j] = 0
                if marks[2 * j + 1] > 1:
                    marks[2 * j + 1] = 1
                if marks[2 * j + 1] < 0:
                    marks[2 * j + 1] = 0

            marks = np.reshape(marks, (-1, 2))
            marks[:, 0] *= w
            marks[:, 1] *= h
            marks[:, 0] += face[0]
            marks[:, 1] += face[1]
            landmark = [[int(marks[i, 0]), int(marks[i, 1])] for i in range(marks.shape[0])]
            landmarks.append(landmark)
        return landmarks

class L106Net96_landmark_detector(landmark_detector):
    def __init__(self):
        self.sym = L106_Net96_v2('test')
        self.pretrained = pretrained_96
        self.epoch = epoch_96
        self.data_size = data_size_96
        self.args, self.auxs = load_param(self.pretrained, self.epoch, convert=True, ctx=ctx)
        self.data_shapes = {'data': (1, 3, self.data_size, self.data_size)}
        # mx.cpu(), x=(5,4),grad_req='null'
        self.executor = self.sym.simple_bind(ctx, grad_req='null', **dict(self.data_shapes))

    def det_landmarks(self, img, faces):
        landmarks = []
        for face in faces:
            image = img[face[1]:face[1]+face[3], face[0]:face[0]+face[2], :]
            h, w, d = image.shape
            image = cv2.resize(image, (self.data_size, self.data_size))
            newimg1 = np.zeros(self.data_shapes['data'], dtype=np.float)
            newimg1[0, 0, :, :] = (image[:, :, 0] - 127.5) * 0.0078125
            newimg1[0, 1, :, :] = (image[:, :, 1] - 127.5) * 0.0078125
            newimg1[0, 2, :, :] = (image[:, :, 2] - 127.5) * 0.0078125
            self.args['data'] = mx.nd.array(newimg1, ctx)
            self.executor.copy_params_from(self.args, self.auxs)
            out_list = [[] for _ in range(len(self.executor.outputs))]
            self.executor.forward(is_train=False)
            for o_list, o_nd in zip(out_list, self.executor.outputs):
                o_list.append(o_nd.asnumpy())
            out = list()
            for o in out_list:
                out.append(np.vstack(o))
            marks = out[0]
            for j in range(int(len(marks) / 2)):
                if marks[2 * j] > 1:
                    marks[2 * j] = 1
                if marks[2 * j] < 0:
                    marks[2 * j] = 0
                if marks[2 * j + 1] > 1:
                    marks[2 * j + 1] = 1
                if marks[2 * j + 1] < 0:
                    marks[2 * j + 1] = 0

            marks = np.reshape(marks, (-1, 2))
            marks[:, 0] *= w
            marks[:, 1] *= h
            marks[:, 0] += face[0]
            marks[:, 1] += face[1]
            landmark = [[int(marks[i, 0]), int(marks[i, 1])] for i in range(marks.shape[0])]
            landmarks.append(landmark)
        return landmarks
