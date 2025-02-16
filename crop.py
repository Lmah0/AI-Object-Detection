import tensorflow as tf
import numpy as np
import cv2
import PIL
from PIL import Image, ImageEnhance
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



def draw_rect(image, box, score):
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
    cv2.putText(image, f"{score:.2f}", (x_center-x_size, y_center-y_size), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)


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

model = tf.keras.models.load_model('Initial.h5')
model.compile()

for file in pathlib.Path('D:/School/SUAV/dataset/Smol').iterdir():

    filepath = f"{file.resolve()}"
    basename = os.path.basename(filepath)

    new_img = tf.keras.utils.load_img(filepath, target_size=(320,320) )
    input_arr = tf.keras.utils.img_to_array(new_img)
    input_arr = np.array([input_arr/128.-1])  # Convert single image to a batch.
    # input_arr = tf.keras.applications.mobilenet.preprocess_input(input_arr)
    predictions = model.predict(input_arr)
    boxes, confidence = decoder(predictions)
    # print(boxes.shape)
    # print(confidence.shape)



    confidence = [c[1] if c[1] > c[0] else 0 for c in confidence[0] ]
    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        boxes[0],
        confidence,
        50,
        iou_threshold=0.5,
        score_threshold=0.6,
        soft_nms_sigma=0.5,
        name=None
    )
    img = Image.open(f"D:/School/SUAV/dataset/Large/{basename}")
    converter = ImageEnhance.Color(img)
    img2 = converter.enhance(3)
    pix = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    # new_img = cv2.resize(pix, (640, 640))
    # print(predictions.shape)

    for i in range(len(selected_indices)):
        box = boxes[0][selected_indices[i]]
        conf = selected_scores[i]
        filename = os.path.splitext(basename)[0]
        cropped = crop(pix, box)
        cv2.imwrite(f"D:/School/SUAV/dataset/Cropped_quant/{filename}_{i}.png", kmeans_color_quantization(cropped))
        #cv2.imshow(f"D:/School/SUAV/dataset/Cropped_quant/{filename}_{i}.png", kmeans_color_quantization(cropped))
        #cv2.waitKey(0)
        # cv2.imwrite(crop(pix, box, conf))

    # cv2.imshow("image", pix)
    # cv2.waitKey(0)
 


