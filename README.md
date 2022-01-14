# Drone-Balloon-Camouflage-RealTime-Detection

**General Description**

Aerostats and camouflaged object detection system consist of 2 separate and totally different object detection and tracking algorithms:

- **Aerostats object detection algorithm**

	The aerostats object detection algorithm is a Real-Time Machine Learning object detection algorithm, based on YoloV5 Pytorch implementation of the YoloV5 algorithm 		family.
	The algorithm is capable of analyzing and inference from multiple input sources, like local image files, video files and webcam (Using an open-source library could OpenCV).
	The objects that are predicted by the algorithm are visualized in a Real-Time stream by bounding boxes that are drawn inside the frame. 

- **Camouflaged object detection algorithm**

	The camouflaged object detection algorithm is based on 13 Haralick Features, which Robert Haralick suggested in his article from 1973, and in a SVM (Support Vector Machine) model.

	The algorithm is capable of analyzing and inference from images and videos.
	The objects that are predicted by the algorithm are visualized in a stream by mask object that is drawn inside the frame, and over the detected object. 
	
**2. How to use**

Inside the project's root folder, 'Balloon-Camouflage-Detection', there is the main GUI component, called 'Demo_Dashboard'.
Edit 'Demo_Dashboard' as follows:

1)	In line 56, change first argument of 'subprocess.call' to the following command, after changing the red paths to the correct paths on your local machine:
`[PYTHON_PATH]/python.exe [PATH_TO_FOLDER]/Drone-Balloon-Camouflage-RealTime-Detection/yolov51/detect.py --source 0 --weights [PATH_TO_FOLDER]/Drone-Balloon-Camouflage-RealTime-Detection/yolov51/best.pt --conf-thres 0.7 --half --dnn`


2) In 'detect.py' inside 'yolov51' subfolder, change line 51 and define the correct COM number for the Arduino USB interface:
`ArduinoSerial=serial.Serial('COM6',9600,timeout=0.1) # Define COM number for arduino interface`
	
3)	Install dependencies:

Make sure 'PIP' package is installed.
Open PowerShell from 'yolov51' folder and execute:
`pip install -r requirements.txt`

4) Run 'Demo_Dashboard.py', and choose the algorithm you want to fire:
![Dashboard](https://eamobileisrael.com/demodash.png)
 

	

