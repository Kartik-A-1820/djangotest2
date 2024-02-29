import pickle
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
import logging

def calc_cm_vals(y_true, y_pred, class_index=None):
    pred = tf.argmax(y_pred, axis=1)
    true = tf.reshape(y_true, (-1,))
    
    tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(true, class_index), 
                                              tf.equal(pred, class_index)), 
                                               tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(true, class_index), 
                                              tf.equal(pred, class_index)), 
                                               tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(true, class_index), 
                                             tf.not_equal(pred, class_index)), 
                                              tf.float32))
    return tp, fp, fn

def recall(y_true, y_pred, class_index=1):
    tp, fp, fn = calc_cm_vals(y_true, y_pred, class_index)
    return tp / (tp + fn + K.epsilon()) 

def precision(y_true, y_pred, class_index=0):
    tp, fp, fn = calc_cm_vals(y_true, y_pred, class_index)
    return tp / (tp + fp + K.epsilon()) 



custom_metrics = {'recall_1':recall, 'precision_0':precision}
# model_filepath = r"C:\Users\KARTIK\Desktop\Mobile_APP\models\sardine.h5"
model_filepath = r"/home/ec2-user/DJANGOTEST/fish_product_models/sardine.h5"

model = None

def preprocess_img(image, target_size=(280, 180)):
    if image is not None:
        height, width, channels = image.shape
        if width<height :
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)                
    preprocessed_image = cv2.resize(image, target_size)
    reshaped_image = preprocessed_image.reshape(1, *preprocessed_image.shape)
    image = reshaped_image.astype(np.float32)
    image = image / 255.0
    return image

def predict_sardine(img):
    global model
    if not model:
        logging.info('LOADING SARDINE MODEL!!!!!')
        try:
            model = load_model(r"/home/ec2-user/ML/fish_product_models/sardine.h5", custom_objects=custom_metrics)
        except:
            model = load_model(r"fish_product_models/sardine.h5", custom_objects=custom_metrics)
        # model_filepath = r"/home/ec2-user/fish_product_models/sardine.h5"
    target_size=(280, 180)
    label_map = {'Bad': 0, 'Good': 1}
    preprocessed_img = preprocess_img(img, target_size)
    prediction = model.predict(preprocessed_img)
    prediction = np.array([np.argmax(pred) for pred in prediction])  
    reverse_label_map = {idx: img_type for img_type, idx in label_map.items()}
    predicted_class = reverse_label_map[prediction[0]] 
    return predicted_class
