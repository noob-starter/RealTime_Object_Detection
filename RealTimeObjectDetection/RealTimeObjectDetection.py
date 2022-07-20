# import the important library 
import numpy as np
import imutils
import cv2
import time


prototxt = "D:\\Study\\AI internship\\Day_11\\RealTimeObjectDetection\\MobileNetSSD_deploy.prototxt.txt"
model = "D:\\Study\\AI internship\\Day_11\\RealTimeObjectDetection\\MobileNetSSD_deploy.caffemodel"
confThresh = 0.2  # check the confidence level object detected 

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor","book"]
# to choose the colors randomly 
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Loading model...")
# loading the caffe file 
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print("Model Loaded")
print("Starting Camera Feed...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
	_,frame = vs.read()
	frame = imutils.resize(frame, width=500)

	(h, w) = frame.shape[:2]
	# to satisfy the prerequest of MobileNetSSD
	imResizeBlob = cv2.resize(frame, (300, 300))
	# to convert the color image frame to blob image format 
	blob = cv2.dnn.blobFromImage(imResizeBlob,0.007843, (300, 300), 127.5)

	net.setInput(blob)
	detections = net.forward()  # returns the accuracy, CLASSES (list made by us)
	detShape = detections.shape[2]
	for i in np.arange(0,detShape):
		# to identify weather there is object present or not 
		confidence = detections[0, 0, i, 2]
		# print(confidence)
		if confidence > confThresh:  
			# to get the index of object observed in frame from the CLASSES list    
			idx = int(detections[0, 0, i, 1])
			print("ClassID:",detections[0, 0, i, 1])

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")  # return value gives the coordinates for the box on object
			if CLASSES[idx] == 'bottle':
				label="I need water"
			else:
				label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)

			cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 4)
			
			if startY - 15 > 15:
				y = startY - 15
			else:
				y = startY + 15
			cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1)
	if key == 27:
		break
vs.release()
cv2.destroyAllWindows()

