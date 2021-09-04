
# import the necessary packages
from tensorflow.python.keras.preprocessing import image as image_utils
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.applications import VGG16
import numpy as np
import cv2

"""
frame_id = 0
frame = 0
cap = cv2.VideoCapture('./video.mov') 
while(1): 
	ret, frame = cap.read() 
	frame_id += 1
	if frame_id>10:
		break
"""
# construct the argument parser and parse the arguments
"""
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())
"""
# load the original image via OpenCV so we can draw on it and display
# it to our screen later
#orig = frame.copy()
# load the input image using the Keras helper utility while ensuring
# that the image is resized to 224x224 pxiels, the required input
# dimensions for the network -- then convert the PIL image to a
# NumPy array
#print("[INFO] loading and preprocessing image...")
#image = cv2.resize(frame, (224,224))
#image = image_utils.load_img(frame, target_size=(224, 224))
#image = image_utils.img_to_array(image)

# our image is now represented by a NumPy array of shape (224, 224, 3),
# assuming TensorFlow "channels last" ordering of course, but we need
# to expand the dimensions to be (1, 3, 224, 224) so we can pass it
# through the network -- we'll also preprocess the image by subtracting
# the mean RGB pixel intensity from the ImageNet dataset
#image = np.expand_dims(image, axis=0)
#image = preprocess_input(image)

# load the VGG16 network pre-trained on the ImageNet dataset
#print("[INFO] loading network...")
#model = VGG16(weights="imagenet")

# classify the image
#breakpoint()
#print("[INFO] classifying image...")
#preds = model.predict(image)
#print("********\n",preds)
#P = decode_predictions(preds,top=20)
#print("********\n",P)
# loop over the predictions and display the rank-5 predictions +
# probabilities to our terminal
"""
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

# load the image via OpenCV, draw the top prediction on the image,
# and display the image to our screen
#orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)


cap.release() 
cv2.destroyAllWindows() 
"""

M = VGG16(weights="imagenet")

def classifyVGG(img, model=M):
	image = cv2.resize(img, (224,224))

	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)
	preds = model.predict(image)
	P = decode_predictions(preds,top=5)
	for (i, (imagenetID, label, prob)) in enumerate(P[0]):
		print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))