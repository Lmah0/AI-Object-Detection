import tensorflow as tf
import numpy as np
import cv2
import pathlib
import sys
import os
import box_utils
from scipy.special import expit
from ssd_config import priors,center_variance,size_variance

def crop(image, box):
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

    x_size = int(abs(x_1-x_2)/2*1.2)
    y_size = int(abs(y_1-y_2)/2*1.2)

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
    cv2.putText(image, f"{text}", (x_center-x_size, y_center-y_size), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)


def decoder(predictions):
    if isinstance(predictions, list):
        predictions = np.concatenate(predictions, axis=0)
    conf = predictions[:, :, :predictions.shape[2] - 4]  # / 256.0  
    locations = predictions[:, :, predictions.shape[2] - 4:]  # / 256.0
    confidences = expit(conf)
    boxes = box_utils.np_convert_locations_to_boxes(locations, priors, center_variance, size_variance)
    boxes = box_utils.np_center_form_to_corner_form(boxes)
    return boxes, confidences

detection_model = tf.keras.models.load_model('Initial.h5')
detection_model.compile()

shapes_model = tf.keras.models.load_model('Shapes.h5')
shape_labels = ["bg", "circle", "cross", "hept", "hex", "oct", "pent", "quart", "rect", "semi", "square", "star", "trap", "tri"]
shapes_model.compile()

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
    img = cv2.imread(f"./images/{basename}")
    new_img = cv2.resize(img, (int(640/img.shape[0]*img.shape[1]), 640))
    # print(predictions.shape)

    for i in range(len(selected_indices)):
        box = boxes[0][selected_indices[i]]
        conf = selected_scores[i]
        crop_img = crop(img, box)
        crop_img = np.array([crop_img])
        input_arr = tf.keras.applications.mobilenet.preprocess_input(crop_img)
        input_arr = tf.image.resize(input_arr, (160,160))
        shapes_predition = shapes_model.predict(input_arr)
        max_class = np.argmax(shapes_predition[0])
        # print(shape_labels[max_class])
        # cv2.imshow("image", crop_img[0])
        # cv2.waitKey(0)
        draw_rect(new_img, box, f"shape: {shape_labels[max_class]}")
    cv2.imshow("image", new_img)
    cv2.waitKey(0)


