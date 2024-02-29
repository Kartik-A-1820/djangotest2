import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
import logging


model = None
target_size = (224, 224)
threshold = 0.5 #0.478963

def process_image(image):
    resized_image = cv2.resize(image,target_size)
    processed_image = tf.keras.applications.densenet.preprocess_input(resized_image)
    processed_image = tf.expand_dims(processed_image, axis=0)
    return processed_image

def predict_mackerel(image):
    global model
    if not model:
        logging.info('LOADING MACKEREL MODEL!!!!!')
        try:
            model = load_model(r"/home/ec2-user/ML/fish_product_models/best_model_f1_(mackerel).h5")
        except:
            model = load_model(r"fish_product_models/best_model_f1_(mackerel).h5")

        # model = load_model(r"C:\Users\KARTIK\Desktop\Mobile_APP\models\best_model_f1_(mackerel).h5")
        # model = load_model(r"/home/ec2-user/fish_product_models/best_model_f1_(mackerel).h5")
    image = process_image(image)
    pred = model.predict(image, verbose=0)
    print(pred[0][0])
    label = 'Good' if pred[0][0] >= threshold else 'Bad'
    return label


# import numpy as np
# import cv2
# import tensorflow as tf
# from keras.models import load_model

# def f1_score_metric(y_true, y_pred):
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(tf.math.round(y_pred), tf.float32)
#     tp = tf.reduce_sum(y_true * y_pred)
#     fp = tf.reduce_sum(y_pred) - tp
#     fn = tf.reduce_sum(y_true) - tp
#     precision = tp / (tp + fp + tf.keras.backend.epsilon())
#     recall = tp / (tp + fn + tf.keras.backend.epsilon())
#     f1_score = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
#     return f1_score

# model = load_model(r"C:\Users\KARTIK\Desktop\Mobile_APP\models\mackerel2.h5", custom_objects={"f1_score_metric":f1_score_metric})
# # model = load_model(r"/home/ec2-user/fish_product_models/mackerel2.h5", custom_objects={"f1_score_metric":f1_score_metric})

# def process_image(image):
#     resized_image = resize_image(image, size=(224, 224))
#     resized_image = cv2.resize(resized_image,(224,224)) #BGR
#     processed_image = resized_image/255.0
#     processed_image = np.expand_dims(processed_image, axis=0)
#     return tf.convert_to_tensor(processed_image, dtype=tf.float32)

# def resize_image(image, size):
#     h, w = image.shape[0],image.shape[1]
#     aspect_ratio = w / h
#     if aspect_ratio > 1:
#         new_w = size[0]
#         new_h = int(new_w / aspect_ratio)
#     else:
#         new_h = size[1]
#         new_w = int(new_h * aspect_ratio)
#     resized_image = cv2.resize(image, (new_w, new_h))
#     padded_image = pad_image(resized_image, size)
#     return padded_image

# def pad_image(image, size):
#     h, w = image.shape[0], image.shape[1]
#     pad_h = (size[1] - h) // 2
#     pad_w = (size[0] - w) // 2
#     padded_image = cv2.copyMakeBorder(
#         image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255)
#     )
#     return padded_image

# def load_image(image):
#     image = np.array(image)
#     processed_image = process_image(image)
#     return processed_image

# def threshold(predictions):
#     return np.where(predictions > 0.55, "Good", "Bad")

# def predict_mackerel(image):
#     image = load_image(image)
#     pred = model.predict(image, verbose=0)
#     label = ['Bad', 'Good']
#     return label[np.argmax(pred[0])]
