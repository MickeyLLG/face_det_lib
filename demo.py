import argparse
import cv2
import time
from det_dlib import dlib_landmark_detector,dlib_face_detector
from det_pfld import pfld_landmark_detector
from det_linzaer import linzaer_face_detector

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference.')
parser.add_argument('--face_det', default='dlib',
                    type=str, help='Method used to detect faces.')
parser.add_argument('--landmark_det', default='dlib',
                    type=str, help='Method used to detect landmarks.')
parser.add_argument("--image", type=str, default=None,
                    help="image file to be processed.")
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
parser.add_argument("--save_path", type=str, default=None,
                    help="The save path.")
parser.add_argument("--time_cost", type=str, default=None,
                    help="The path(.txt) to save time cost.")
args = parser.parse_args()

if __name__=='__main__':
    face_detector=None
    landmark_detector=None
    if args.face_det=='dlib':
        face_detector=dlib_face_detector()
    elif args.face_det=='linzaer':
        face_detector=linzaer_face_detector()
    else:
        print("Don't support face detector!")
        exit(0)
    if args.landmark_det=='dlib':
        landmark_detector=dlib_landmark_detector()
    elif args.landmark_det=='pfld':
        landmark_detector=pfld_landmark_detector()
    else:
        print("Don't support landmark detector!")
        exit(0)
    save_path=args.save_path
    time_cost=[]
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
        fps = 15
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(save_path, fourcc, fps, size)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            tic=time.time()
            faces=face_detector.det_faces(frame)
            landmarks=landmark_detector.det_landmarks(frame,faces=faces)
            time_cost.append(time.time()-tic)
            for face in faces:
                cv2.rectangle(frame,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(0,255,0),1)
            for marks in landmarks:
                for mark in marks:
                    cv2.circle(frame, (mark[0], mark[1]), 1, (255, 0, 0), 1)
            out.write(frame)
            cv2.imshow('video', frame)
            k = cv2.waitKey(40)
            # q键退出
            if (k & 0xff == ord('q')):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        frame=cv2.imread(args.image)
        tic=time.time()
        faces = face_detector.det_faces(frame)
        landmarks = landmark_detector.det_landmarks(frame, faces=faces)
        time_cost.append(time.time()-tic)
        for face in faces:
            cv2.rectangle(frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 1)
        for marks in landmarks:
            for mark in marks:
                cv2.circle(frame, (mark[0], mark[1]), 1, (255, 0, 0), 1)
        cv2.imshow('image',frame)
        cv2.waitKey()
        cv2.imwrite(save_path, frame)
    if args.time_cost is not None:
        fw = open(args.time_cost, 'w')
        fw.write('\n'.join(time_cost))
        fw.close()










