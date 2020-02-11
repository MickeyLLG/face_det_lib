import mxnet as mx
import numpy as np
import cv2
import sys
import time
from mxnet import nd

sys.path.append('det_mobileface/MobileFace_Detection/')
sys.path.append('det_mobileface/MobileFace_Detection/utils/')
sys.path.append('det_mobileface/MobileFace_Tracking/')
from mobilefacedetnet import mobilefacedetnet_v2
from mobileface_sort_v1 import Sort
from data_presets import data_trans
from Interface import face_detector

model = 'det_mobileface/MobileFace_Detection/model/mobilefacedet_v2_gluoncv.params'  # Pretrained model path.
sort_max_age = 10  # Threshold of object score when visualize the bboxes.
sort_min_hits = 3  # Threshold of object score when visualize the bboxes.
thresh = 0.5  # Threshold of object score when visualize the bboxes.

# context list
ctx = [mx.cpu()]
net = mobilefacedetnet_v2(model)
net.set_nms(0.45, 200)
net.collect_params().reset_ctx(ctx=ctx)
mot_tracker = Sort(sort_max_age, sort_min_hits)
img_short = 256
colors = np.random.rand(32, 3) * 255


class mobileface_face_detector(face_detector):
    def det_faces(self, img):
        dets = []
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_nd = nd.array(frame_rgb)
        x, image = data_trans(frame_nd, short=img_short)
        x = x.as_in_context(ctx[0])
        # ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
        tic = time.time()
        # pic=np.reshape(x.asnumpy(),(256,455,3))*255
        # cv2.imwrite('1.jpg',pic)
        result = net(x)
        toc = time.time() - tic
        # print('Detection inference time:%fms' % (toc * 1000))

        ids, scores, bboxes = [xx[0].asnumpy() for xx in result]  # 这句话里的asnumpy函数耗时特别多
        h, w, c = img.shape
        scale = float(img_short) / float(min(h, w))

        for i, bbox in enumerate(bboxes):
            if scores[i] < thresh:
                continue
            xmin, ymin, xmax, ymax = [int(x / scale) for x in bbox]
            # result = [xmin, ymin, xmax, ymax, ids[i], scores[i]]
            result = [xmin, ymin, xmax, ymax, ids[i]]
            dets.append(result)

        dets = np.array(dets)
        tic = time.time()
        trackers = mot_tracker.update(dets)
        toc = time.time() - tic
        # print('Tracking time:%fms' % (toc * 1000))
        faces = []
        for d in trackers:
            faces.append([int(d[0]), int(d[1]), int(d[2] - d[0]), int(d[3] - d[1])])
        return faces
