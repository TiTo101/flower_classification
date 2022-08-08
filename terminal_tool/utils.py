import tensorflow as tf
import numpy as np
import json
from PIL import Image
from heapq import nlargest

def process_image(image_array, size=224):
    """
    processes image arrays to 
    1) be of the sames size
    2) normalize pixzel values
    """
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    image_tensor = tf.image.resize(image_tensor, (size, size))
    image_tensor /= 255
    image_array = image_tensor.numpy()
    return image_array

def predict(image_path, 
            model, 
            top_k=5, 
            classes_json_path='./label_map.json'):
    """
    predicting the top_k flowers that might be shown
    in a given picture
    """
    with open(classes_json_path, 'r') as f:
        class_names = json.load(f)
    image =  Image.open(image_path)
    image_array = np.asarray(image)
    image_array = process_image(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    probs = model.predict(image_array)
    class_names_ls = list(class_names.values())
    guesses_dict = dict(zip(class_names_ls, probs[0]))
    guesses_dict = dict(sorted(guesses_dict.items(), key=lambda item: item[1]))
    classes = nlargest(top_k, guesses_dict, key=guesses_dict.get)
    probs = [guesses_dict[x] for x in classes]
    return probs, classes