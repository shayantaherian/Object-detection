import cv2
import numpy as np
import imutils
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_cfg', type = str, default = '',
                    help = 'Path to config file')
parser.add_argument('--model_weights', type=str,
                    default='',
                    help='path to weights of model')
parser.add_argument('--video', type=str, default='',
                    help='path to video file')
parser.add_argument('--src', type=int, default=0,
                    help='source of the camera')
parser.add_argument('--output_dir', type=str, default='outputs/',
                    help='path to the output directory')
args = parser.parse_args()

# print the arguments
print('----- info -----')
print('[i] The config file: ', args.model_cfg)
print('[i] The weights of model file: ', args.model_weights)
print('[i] Path to video file: ', args.video)
print('###########################################################\n')

frameWidth= 840
frameHeight = 680

# Load YOLO
net = cv2.dnn.readNet(args.model_weights, args.model_cfg)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()] # we put the names in to an array

layers_names = net.getLayerNames()
output_layers = [layers_names[i[0] -1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size = (len(classes), 3))

# If using webcame
#cameraFeed = False
#if cameraFeed:
   # video = cv2.VideoCapture(0)

output_file = ''
video = cv2.VideoCapture(args.video)
output_file = args.video[:-4].rsplit('/')[-1] + '_Car.mp4'



fourcc = cv2.VideoWriter_fourcc('C','Y','U','V')
video_writer = cv2.VideoWriter('output.mov', fourcc, 20.0, (640,480))


#video = cv2.VideoCapture(args.video)
font = cv2.FONT_HERSHEY_PLAIN
frame_id = 0
starting_time = time.time()
while True:
    has_frame, frame = video.read()
    frame = cv2.resize(frame, (frameWidth, frameHeight), None)

    if not has_frame:
        print('[i] ==> Done processing!!!')
        print('[i] ==> Output file is stored at', os.path.join(args.output_dir, output_file))
        cv2.waitKey(1000)
        break

    frame_id += 1
    height, width, channels = frame.shape
    # Detect image
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0,0,0), swapRB = True, crop = False)
    net.setInput(blob)
    start = time.time()
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y -h / 2)
                #cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0))

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                # Name of the object
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "{}: {:.2f}%".format(classes[class_ids[i]], confidences[i]*100)
            color = colors[i]
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x,y+10), font, 1, color, 1)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS:" + str(fps), (10,30), font, 2, (0, 0, 0), 1)

    video_writer.write(frame)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
print('==> All done!')
print('***********************************************************')
