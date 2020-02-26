import argparse
import cv2
import time
import os


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--face_det', default='centerface',
                    type=str, help='Method used to detect faces.')
parser.add_argument('--landmark_det', default='frda',
                    type=str, help='Method used to detect landmarks.')
parser.add_argument("--image", type=str, default=None,
                    help="image file to be processed.")
parser.add_argument("--video", type=str, default='test/test_15fps.avi',
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
parser.add_argument("--save_path", type=str, default='result/result.avi',
                    help="The save path.")
parser.add_argument("--fps", type=int, default=15,
                    help="The frames per second of the video to save.")
parser.add_argument("--time_cost", type=str, default=None,
                    help="The path(.txt) to save time cost per frame.")
parser.add_argument("--stretchY", type=float, default=1.1,
                    help="The face boxes usually need to be stretched along axis Y,this is the stretch rate.")
args = parser.parse_args()

if __name__ == '__main__':
    # Generate detectors.
    face_detector = None
    landmark_detector = None
    if args.face_det == 'dlib':
        from det_dlib import dlib_face_detector
        face_detector = dlib_face_detector()
    elif args.face_det == 'mtcnn':
        from det_mtcnn import mtcnn_face_detector
        face_detector = mtcnn_face_detector()
    elif args.face_det == 'linzaer':
        from det_linzaer import linzaer_face_detector
        face_detector = linzaer_face_detector()
    elif args.face_det == 'centerface':
        from det_centerface import centerface_face_detector
        face_detector = centerface_face_detector()
    elif args.face_det == 'biubug':
        from det_biubug import biubug_face_detector
        face_detector = biubug_face_detector()
    elif args.face_det == 'mobileface':
        from det_mobileface import mobileface_face_detector
        face_detector = mobileface_face_detector()
    elif args.face_det == 'zqmtcnn':
        from det_zqcnn import zqmtcnn_face_detector
        face_detector = zqmtcnn_face_detector()
    else:
        print("Don't support face detector!")
        exit(0)
    if args.landmark_det == 'dlib':
        from det_dlib import dlib_landmark_detector
        landmark_detector = dlib_landmark_detector()
    elif args.landmark_det == 'pfld':
        from det_pfld import pfld_landmark_detector
        landmark_detector = pfld_landmark_detector()
    elif args.landmark_det == 'L106Net112':
        from det_zqcnn import L106Net112_landmark_detector
        landmark_detector = L106Net112_landmark_detector()
    elif args.landmark_det == 'L106Net96':
        from det_zqcnn import L106Net96_landmark_detector
        landmark_detector = L106Net96_landmark_detector()
    elif args.landmark_det == 'cnn':
        from det_cnn import cnn_landmark_detector
        landmark_detector = cnn_landmark_detector()
    elif args.landmark_det == 'frda':
        from det_frda import frda_landmark_detector
        landmark_detector = frda_landmark_detector()
    else:
        print("Don't support landmark detector!")
        exit(0)

    # Make dirs.
    save_path = args.save_path
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # Detection and visualization.
    time_cost = []
    if args.image is None:
        video_src = args.cam if args.cam is not None else args.video
        if video_src is None:
            exit(0)
        if args.save_path is None:
            print("Save path is required!")
            exit(0)
        cap = cv2.VideoCapture(video_src)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # fps = cap.get(cv2.CAP_PROP_FPS)
        fps = args.fps
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(save_path, fourcc, fps, size)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            tic = time.time()
            faces = face_detector.det_faces(frame)  # faces detection
            faces = [[face[0], face[1], face[2], int(face[3] * args.stretchY)] for face in faces]  # stretch along Y
            landmarks = landmark_detector.det_landmarks(frame, faces=faces)  # landmarks detection
            time_cost.append(str(time.time() - tic))
            for face in faces:
                cv2.rectangle(frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 1)
            for marks in landmarks:
                for mark in marks:
                    cv2.circle(frame, (mark[0], mark[1]), 1, (255, 0, 0), 1)
            out.write(frame)
            cv2.imshow('video', frame)
            k = cv2.waitKey(40)
            # q键退出
            if k & 0xff == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        if args.save_path is None:
            print("Save path is required!")
            exit(0)
        frame = cv2.imread(args.image)
        tic = time.time()
        faces = face_detector.det_faces(frame)
        landmarks = landmark_detector.det_landmarks(frame, faces=faces)
        time_cost.append(time.time() - tic)
        for face in faces:
            cv2.rectangle(frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 1)
        for marks in landmarks:
            for mark in marks:
                cv2.circle(frame, (mark[0], mark[1]), 1, (255, 0, 0), 1)
        cv2.imshow('image', frame)
        cv2.waitKey()
        cv2.imwrite(save_path, frame)
    if args.time_cost is not None:
        fw = open(args.time_cost, 'w')
        fw.write('\n'.join(time_cost))
        fw.close()
