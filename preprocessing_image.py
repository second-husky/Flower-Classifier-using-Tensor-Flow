import numpy as np
import tensorflow as tf
def process_image(image):
    image_size = 224
    image = tf.cast(image,tf.float32)
    #image = tf.convert_to_tensor(image, dtype = tf.float32)
    image = tf.image.resize(image,(image_size,image_size))
    image /= 255
    return image.numpy()

