import sys
import time
import mxnet as mx

sys.path.append('det_zqcnn')
from tools.load_model import load_param
from core.symbol import P_Net20, R_Net, O_Net
from core.fcn_detector import FcnDetector
from core.detector import Detector
from core.MtcnnDetector20 import MtcnnDetector
from Interface import face_detector


def get_face_boxes(img, prefix, epoch, batch_size, ctx,
                   thresh=[0.6, 0.6, 0.7], min_face_size=24):
    detectors = [None, None, None]

    # load pnet model
    args, auxs = load_param(prefix[0], epoch[0], convert=True, ctx=ctx)
    PNet = FcnDetector(P_Net20("test"), ctx, args, auxs)
    detectors[0] = PNet

    # load rnet model
    args, auxs = load_param(prefix[1], epoch[1], convert=True, ctx=ctx)
    RNet = Detector(R_Net("test"), 24, batch_size[1], ctx, args, auxs)
    detectors[1] = RNet

    # load onet model
    args, auxs = load_param(prefix[2], epoch[2], convert=True, ctx=ctx)
    ONet = Detector(O_Net("test", False), 48, batch_size[2], ctx, args, auxs)
    detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, ctx=ctx, min_face_size=min_face_size,
                                   stride=4, threshold=thresh, slide_window=False)

    t1 = time.time()
    # print 'hello1'
    boxes, boxes_c = mtcnn_detector.detect_pnet20(img)
    # print 'hello2'
    boxes, boxes_c = mtcnn_detector.detect_rnet(img, boxes_c)
    # print 'hello3'
    boxes, boxes_c = mtcnn_detector.detect_onet(img, boxes_c)

    # print 'time: ',time.time() - t1
    return boxes_c


prefix = ['det_zqcnn/model/pnet20_v1', 'det_zqcnn/model/rnet_v1', 'det_zqcnn/model/onet_v1']
ctx = mx.cpu(0)
epoch = [16, 16, 16]
batch_size = [2048, 256, 16]
thresh = [0.5, 0.5, 0.7]
min_face = 20


class zqmtcnn_face_detector(face_detector):
    def det_faces(self, img):
        try:
            boxes = get_face_boxes(img, prefix, epoch, batch_size, ctx, thresh, min_face)
            faces = []
            for b in boxes:
                x, y, r, b = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                faces.append([x, y, r - x, b - y])
            return faces
        except:
            return []
