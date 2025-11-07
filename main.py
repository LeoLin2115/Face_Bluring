import cv2 as cv
import mediapipe as mp
import os
import argparse
import sys

def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    # print(out.detections)
    if out.detections is not None:
        for detection in out.detections:
            location = detection.location_data
            bbox = location.relative_bounding_box
            print(type(bbox), bbox)
            x1, y1, w, h = max(0, bbox.xmin), max(0,bbox.ymin), bbox.width, bbox.height
            x1, y1, w, h = int(x1 * W), int(y1 * H), int(w * W), int(h * H)
            # img = cv.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 3)

            img[y1:y1 + h, x1:x1 + w] = cv.blur(img[y1:y1 + h, x1:x1 + w], (50, 50))
    return img

args = argparse.ArgumentParser()
# args.add_argument('--mode', default = 'image')
# args.add_argument('--filePath', default = './data/testImage.jpg')
# args.add_argument('--mode', default = 'video')
# args.add_argument('--filePath', default = './data/testvideo.mp4')
args.add_argument('--mode', default = 'camera')
args.add_argument('--filePath', default = None)
args = args.parse_args()

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# read image
# img_path = './data/testImage.jpg'
# img = cv.imread(img_path)
# if img is None:
#     raise FileNotFoundError(f"Could not read image at {os.path.abspath(img_path)}. Check path, filename, and permissions.")

# detect image
mp_face_detection = mp.solutions.face_detection



with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as face_detection:
    if args.mode in ['image']:
        img = cv.imread(args.filePath)
        img = process_img(img, face_detection)
        cv.imwrite(os.path.join(output_dir, 'output.jpg'), img)

    elif args.mode in ['video']:
        cap = cv.VideoCapture(args.filePath)
        ret, frame = cap.read()
        output_video = cv.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                      cv.VideoWriter_fourcc(*'mp4v'),
                                      20, (frame.shape[1], frame.shape[0]))
        
        while ret:
            frame = process_img(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()
            # cv.imshow('frame', frame)
            # if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        output_video.release()
        cv.destroyAllWindows()
    elif args.mode in ['camera']:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Camera not opened. Check device index or drivers.")
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to grab frame from camera.")
        while ret:
            frame = process_img(frame, face_detection)
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = cap.read()
        cap.release()
        cv.destroyAllWindows()

# blur image

# cv.imshow('img', img)
# cv.waitKey(0)

# save image

print("sys.executable:", sys.executable)
print("sys.version:", sys.version.splitlines()[0])
print("cwd:", os.getcwd())
print("__file__:", os.path.abspath(__file__))
print("dir listing:", os.listdir('.'))


