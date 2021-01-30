import argparse
import numpy as np
import preprocessing_image as pi
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import json
from PIL import Image

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = pi.process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, axis = 0)
    preds = model.predict(processed_test_image)
    probs = (-np.sort(-preds)[:,:top_k])[0].tolist()
    classes = ((np.argsort((-1)*preds)[:,:top_k])[0]+1).astype(str).tolist()
    
    return probs, classes

parser = argparse.ArgumentParser(
    description = 'This is a python program to predict the name of flowers by images'
)
parser.add_argument('image_path', action = "store"，help = "image_path", default = "./test_images/hard-leaved_pocket_orchid.jpg")
parser.add_argument('saved_model', action = "store", help = "model for classfier", default = "my_model.h5")
parser.add_argument('--top_k', type = int, action = "store", dest = 'top_k', help = "number k of top predictions needed")
parser.add_argument('--category_names', action = "store", dest = 'category_decoder', help = "map for decoding flower names")
args = parser.parse_args()

image_path = args.image_path
reloaded_keras_classfier = tf.keras.models.load_model(args.saved_model, custom_objects={'KerasLayer':hub.KerasLayer})
top_k = args.top_k
category_names = args.category_decoder

if top_k is None:
    top_k = 1
 
probs, classes = predict(image_path,reloaded_keras_classfier,top_k)
if category_names is not None:
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    print("The top {} possible classes of this flower and their possibilities are:".format(top_k))
    for i in range(len(classes)):
        print(class_names[classes[i]],": ", probs[i])
else:
    print("The top {} possible labels of this flower and their possibilities are:".format(top_k))
    for i in range(len(classes)):
        print(classes[i],": ", probs[i])
    
