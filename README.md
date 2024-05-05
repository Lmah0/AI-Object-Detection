# AI-Object-Detection

Convolutional neural network trained for object detection drone targeting. Utilizes single shot architecture with MobileNetV3 for fast processing. Adjusted to custom testing data and local dependencies.

COMMON ERROR: Compiled h(5) files wil only work with older versions of TensorFlow (i.e. tensorflow==2.12.0). Users may need to remove newer versions of TensorFlow to replace with an older version (otherwise model binaries will appear as corrupt).

Drag any of the testing data (data folder) into the images folder to test. Run "runDetections.py" for object detections. 

Development and testing project tuned To local dependencies before competition.