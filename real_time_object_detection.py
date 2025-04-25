from imutils.video import VideoStream
import numpy as np
import sys
import argparse
import imutils
import urllib.request
import time
import cv2
from urllib.request import urlopen

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("--source", required=True,
                help="Source of video stream (webcam/host)")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print(" ______________________")
print("|                      |")
print("| WELCOM SIR           |")
print("| LOADING MODEL....... |")
print("|______________________|")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream
# initialize the video stream
if args["source"] == "webcam":
    vs = cv2.VideoCapture(0)
elif args["source"] == "edison":
    vs = cv2.VideoCapture(url)

# allow the camera sensor to warm up
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the video stream and resize it
    ret, frame = vs.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=800)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                                          confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # show the output frame
    cv2.imshow("Frame", frame)  

    # check for key press
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # Check if the window was closed
    if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break
    if key ==27:
        break

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()

