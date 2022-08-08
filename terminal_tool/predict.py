import numpy as np
#import pandas as pd
#import json
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_hub as hub

from utils import process_image, predict

model_path_root = './my_image_clasifier.h5'
reloaded_keras_model = tf.keras.models.load_model( #tf.
    model_path_root,
    compile=False,
    custom_objects={'KerasLayer':hub.KerasLayer}
)

parser = argparse.ArgumentParser(description='predicts flowers from images')
image_path = parser.add_argument('--f', action="store", dest="image_path", required=True, metavar="FILE")
parser.add_argument('--top_k', action="store", dest="top_k", type=int)
parser.add_argument('--category_names', action="store", dest="category_names")

user_input = parser.parse_args()
image_path = user_input.image_path
top_k = user_input.top_k
classes_json_path = user_input.category_names
if top_k is None:
    top_k = 3
    
if classes_json_path is None:
    classes_json_path = './label_map.json'

ps, classes = predict(image_path, 
                      reloaded_keras_model, 
                      top_k=top_k, 
                      classes_json_path=classes_json_path)
for i in range(top_k):
    class_i = classes[i:i+1][0]
    ps_i = round(ps[i:i+1][0],4)
    print(f"The {i+1}. guess of the flower on the image is a {class_i} with a probability of {ps_i:.4f}")
