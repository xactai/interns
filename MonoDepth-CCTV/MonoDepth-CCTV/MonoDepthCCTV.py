#run ----->     python3 MonoDepthLive.py 

#!/usr/bin/env python
# coding: utf-8
#Imports

import sys
import time
from pathlib import Path

import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import (
    HTML,
    FileLink,
    Pretty,
    ProgressBar,
    Video,
    clear_output,
    display,
)
from openvino.runtime import Core
sys.path.append("../utils")
from notebook_utils import load_image


#Settings

DEVICE = "CPU"
MODEL_FILE = "model/MiDaS_small.xml"

model_xml_path = Path(MODEL_FILE)


#Functions

def normalize_minmax(data):
    #Normalizes the values in `data` between 0 and 1
    return (data - data.min()) / (data.max() - data.min())


def convert_result_to_image(result, colormap="viridis"):
    cmap = matplotlib.cm.get_cmap(colormap)
    result = result.squeeze(0)
    result = normalize_minmax(result)
    result = cmap(result)[:, :, :3] * 255
    result = result.astype(np.uint8)
    return result


def to_rgb(image_data) -> np.ndarray:
    #Convert image_data from BGR to RGB
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)


#Load the Model

ie = Core()
ie.set_property({'CACHE_DIR': '../cache'})
model = ie.read_model(model_xml_path)
compiled_model = ie.compile_model(model=model, device_name=DEVICE)

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)

network_input_shape = list(input_key.shape)
network_image_height, network_image_width = network_input_shape[2:]


#Monodepth on Image

IMAGE_FILE = "data/input_image.png"
image = load_image(path=IMAGE_FILE)
resized_image = cv2.resize(src=image, dsize=(network_image_height, network_image_width))
input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)


#Do inference on the image

result = compiled_model([input_image])[output_key]
result_image = convert_result_to_image(result=result)
result_image = cv2.resize(result_image, image.shape[:2][::-1])


#Display monodepth image


fig, ax = plt.subplots(1, 2, figsize=(20, 15))
ax[0].imshow(to_rgb(image))
ax[1].imshow(result_image);


#Monodepth on Video

#VIDEO_FILE = "Paste your rtsp link"
VIDEO_FILE = "data/sample.mp4"
ADVANCE_FRAMES = 2
SCALE_OUTPUT = 0.5
FOURCC = cv2.VideoWriter_fourcc(*"vp09")


#Load the Video

reader = cv2.VideoCapture(VIDEO_FILE)
width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))


#Do Inference on a Video and Create Monodepth Video


def to_rgb_video(video_data) -> np.ndarray:
    resized_video = cv2.resize(src=video_data, dsize=(network_image_height, network_image_width))
    input_video = np.expand_dims(np.transpose(resized_video, (2, 0, 1)), 0)
    result = compiled_model([input_video])[compiled_model.output(0)]
    result_video = convert_result_to_image(result=result)
    result_video = cv2.resize(result_video, image.shape[:2][::-1])
    return cv2.applyColorMap(result_video, cv2.COLORMAP_VIRIDIS)


#Display Monodepth Video

while True:
    ret, frame = reader.read()
    frame = cv2.resize(frame, (width//4, height//4))
    cv2.imshow("MonoDepth Video", np.hstack((frame, cv2.resize(to_rgb_video(frame), (width//4, height//4)))))
    cv2.waitKey(1)
