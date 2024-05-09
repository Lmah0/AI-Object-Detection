import tensorflow as tf
import numpy as np
import cv2
import pathlib
from PIL import Image, ImageEnhance
import sys
import os
import box_utils
import ShapeLetterColor
import Meta_Data
from scipy.special import expit
from ssd_config import priors,center_variance,size_variance

def crop(image, box, expand_box=1.2):
    # print(image.shape)
    # input()
    height = image.shape[0]
    width = image.shape[1]
    y_1 = box[0] * height
    x_1 = box[1] * width
    y_2 = box[2] * height
    x_2 = box[3] * width
    x_center = int((x_1+x_2)/2)
    y_center = int((y_1+y_2)/2)

    x_size = int(abs(x_1-x_2)/2*expand_box)
    y_size = int(abs(y_1-y_2)/2*expand_box)

    x_min = max(x_center-x_size, 0)
    x_max = min(x_center+x_size, width-1)
    y_min = max(y_center-y_size, 0)
    y_max = min(y_center+y_size, height-1)
    # print(f"{(y_min,y_max, x_min,x_max)=}")
    cropped = image[y_min:y_max, x_min:x_max]
    return cropped

def draw_rect(image, box, text):
    # print(image.shape)
    # input()
    height = image.shape[0]
    width = image.shape[1]
    y_1 = box[0] * height
    x_1 = box[1] * width
    y_2 = box[2] * height
    x_2 = box[3] * width
    x_center = int((x_1+x_2)/2)
    y_center = int((y_1+y_2)/2)

    x_size = int(abs(x_1-x_2)/2)
    y_size = int(abs(y_1-y_2)/2)
    # draw a rectangle on the image
    cv2.rectangle(image, (x_center-x_size, y_center-y_size), (x_center+x_size, y_center+y_size), (255, 255, 255), 1)
    y = y_center-y_size
    for t in text:
        cv2.putText(image, f"{t}", (x_center-x_size, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        y += 16


def decoder(predictions):
    if isinstance(predictions, list):
        predictions = np.concatenate(predictions, axis=0)
    conf = predictions[:, :, :predictions.shape[2] - 4]  # / 256.0  
    locations = predictions[:, :, predictions.shape[2] - 4:]  # / 256.0
    confidences = expit(conf)
    boxes = box_utils.np_convert_locations_to_boxes(locations, priors, center_variance, size_variance)
    boxes = box_utils.np_center_form_to_corner_form(boxes)
    return boxes, confidences

# From https://stackoverflow.com/a/59978096
def kmeans_color_quantization(image, clusters=12, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

detection_model = tf.keras.models.load_model('Initial.h5')
detection_model.compile()

shapes_model = tf.keras.models.load_model('Shapes.h5')
shapes_model.compile()
shape_labels = ["bg", "circle", "cross", "hept", "hex", "oct", "pent", "quart", "rect", "semi", "square", "star", "trap", "tri"]


letter_model = tf.keras.models.load_model('Colored_Letters.h5') 
letter_model.compile()
letter_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Did Not Find"]
DID_NOT_FIND = len(letter_labels)-1

letter_color_model = tf.keras.models.load_model('Color_of_Letter.h5') 
letter_color_model.compile()
bg_color_model = tf.keras.models.load_model('Color_of_BG.h5') 
bg_color_model.compile()

color_name = ["black", "blue",  "brown", "gray", "green", "orange", "purple", "red", "white", "yellow", "NONE"]
# color_name = ["==========", "=====", "yellow/red","=====", "green","=====", "======", "purple", "white", "=====", "======"]

for file in pathlib.Path('./images').iterdir():

    filepath = f"{file.resolve()}"
    basename = os.path.basename(filepath)
    import time
    t = time.time()
    new_img = tf.keras.utils.load_img(filepath, target_size=(320,320))
    input_arr = tf.keras.utils.img_to_array(new_img)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    input_arr = tf.keras.applications.mobilenet.preprocess_input(input_arr)
    predictions = detection_model.predict(input_arr)
    boxes, confidence = decoder(predictions)
    print(boxes.shape)
    print(confidence.shape)


    # background, object
    # confidence = [c[1] if c[1] > c[0] else 0 for c in confidence[0]]
    confidence = [c[1] for c in confidence[0]]
    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        boxes[0],
        confidence,
        100,
        iou_threshold=0.5,
        score_threshold=0.5,
        soft_nms_sigma=0.5,
        name=None
    )
    print(f"Time taken to detect: {time.time()-t} seconds")
    filename = f"C:/Users/schul/Desktop/Testing/AI-Object-Detection/images/{basename}"
    img = Image.open(filename)
    # converter = ImageEnhance.Color(img)
    # img2 = converter.enhance(3)
    # pix = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    pix = np.array(img)
    new_img = cv2.resize(pix, (int(640/pix.shape[0]*pix.shape[1]), 640))
    # print(predictions.shape)
    
    for i in range(len(selected_indices)):
        box = boxes[0][selected_indices[i]]
        conf = selected_scores[i]
        crop_img = crop(pix, box, expand_box=1.1)
        crop_img = kmeans_color_quantization(crop_img)
        input_arr = tf.keras.applications.mobilenet.preprocess_input(np.array([crop_img]))
        input_arr160 = tf.image.resize(input_arr, (160,160))
        shapes_predition = shapes_model.predict(input_arr160)
        shapes_max_class = np.argmax(shapes_predition[0])


        crop_img = crop(crop_img, [0,0,1,1], expand_box=0.55)

        input_arr = tf.keras.applications.mobilenet.preprocess_input(np.array([crop_img]))
        input_arr64 = tf.image.resize(input_arr, (64,64))
        letter_predition = letter_model.predict(input_arr64)
        letter_max_class = np.argmax(letter_predition[0]) 

        crop_img = crop(crop_img, [0,0,1,1], expand_box=0.7)
        input_arr = tf.keras.applications.mobilenet.preprocess_input(np.array([crop_img]))
        input_arr64 = tf.image.resize(input_arr, (64,64))

        letter_color_predition = letter_color_model.predict(input_arr64)
        letter_color_max_class = np.argmax(letter_color_predition[0]) 

        bg_color_predition = bg_color_model.predict(input_arr64)
        bg_color_max_class = np.argmax(bg_color_predition[0])

        print(f"Shape:  {shape_labels[shapes_max_class]} {color_name[bg_color_max_class]}",
              f"Letter:  {letter_labels[letter_max_class]} {color_name[letter_color_max_class]}", sep="\n")
     
        # print(box)
        x_center = (box[1]+box[3])/2
        y_center = (box[0]+box[2])/2
        longitude, latitude = Meta_Data.getGPSPosition(filename, (x_center, y_center))
        # print(f"Shape: {avgShapeHSV}={shapeCol}\tLetter: {avgLetterHSV}={letterCol}")
        draw_rect(new_img, box, [f"Shape:  {shape_labels[shapes_max_class]} {color_name[bg_color_max_class]}",
                                 f"Letter:  {letter_labels[letter_max_class]} {color_name[letter_color_max_class]}",
                                 f"GPS: {longitude:.6f}, {latitude:.6f}"])
    cv2.imshow("image", new_img)
    cv2.waitKey(0)
    cv2.imwrite(f"./images_detect/{basename}", cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
    del img
    del pix

