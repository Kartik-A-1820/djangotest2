import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
import logging

model = None

target_size = (224, 224)
threshold = 0.5

def process_image(image):
    resized_image = cv2.resize(image,target_size)
    processed_image = tf.keras.applications.densenet.preprocess_input(resized_image)
    processed_image = tf.expand_dims(processed_image, axis=0)
    return processed_image

def predict_prawn(image):
    global model
    if not model:
        logging.info('LOADING MACKEREL MODEL!!!!!')
        try:
            model = load_model(r"/home/ec2-user/ML/fish_product_models/wp2.h5")
        except:
            model = load_model(r"fish_product_models/wp2.h5")
        # model = load_model(r"C:\Users\KARTIK\Desktop\Mobile_APP\models\wp.h5")
        # model = load_model(r"/home/ec2-user/fish_product_models/wp2.h5")   
    image = process_image(image)
    pred = model.predict(image, verbose=0)
    print(pred[0][0])
    label = 'Good' if pred[0][0] >= threshold else 'Bad'
    return label


# import cv2
# import numpy as np
# from keras.models import load_model
# import tensorflow as tf
# import keras.backend as K
# from keras.losses import sparse_categorical_crossentropy 

# def calc_cm_vals(y_true, y_pred, class_index=None):
#     pred = tf.argmax(y_pred, axis=1)
#     true = tf.reshape(y_true, (-1,))
    
#     tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(true, class_index), 
#                                               tf.equal(pred, class_index)), 
#                                                tf.float32))
#     fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(true, class_index), 
#                                               tf.equal(pred, class_index)), 
#                                                tf.float32))
#     fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(true, class_index), 
#                                              tf.not_equal(pred, class_index)), 
#                                               tf.float32))
#     return tp, fp, fn

# def recall_0(y_true, y_pred, class_index=0):
#     tp, fp, fn = calc_cm_vals(y_true, y_pred, class_index)
#     return tp / (tp + fn + K.epsilon()) 

# def precision_0(y_true, y_pred, class_index=0):
#     tp, fp, fn = calc_cm_vals(y_true, y_pred, class_index)
#     return tp / (tp + fp + K.epsilon())  
    
# def precision_1(y_true, y_pred, class_index=1):
#     tp, fp, fn = calc_cm_vals(y_true, y_pred, class_index)
#     return tp / (tp + fp + K.epsilon()) 

# def recall_1(y_true, y_pred, class_index=1):
#     tp, fp, fn = calc_cm_vals(y_true, y_pred, class_index)
#     return tp / (tp + fn + K.epsilon()) 

# def precision_0_loss(y_true, y_pred):
#     precision = precision_0(y_true, y_pred)  
#     ce_loss = sparse_categorical_crossentropy(y_true, y_pred, 
#                                         from_logits=False, axis=-1)
#     alpha = 0.001
#     combined_loss = alpha * ce_loss + (1 - alpha) * (1-precision)
#     return combined_loss 

# def precision_1_loss(y_true, y_pred):
#     precision = precision_1(y_true, y_pred)  
#     ce_loss = sparse_categorical_crossentropy(y_true, y_pred, 
#                                               from_logits=False, axis=-1)
#     alpha = 0.001
#     combined_loss = alpha * ce_loss + (1 - alpha) * (1-precision)
#     return combined_loss 



# custom_metrics = {'recall_1':recall_1, 'precision_1':precision_1, 
#                       'recall_0':recall_0, 'precision_0':precision_0,
#                       'precision_0_loss':precision_0_loss,
#                       'precision_1_loss':precision_1_loss}
# model_filepath = r"C:\Users\KARTIK\Desktop\Mobile_APP\models\wp67.h5"
# # model_filepath = r"/home/ec2-user/fish_product_models/wp67.h5"

# model = load_model(model_filepath, custom_objects=custom_metrics)

# def preprocess_img(image, target_size=(280, 180)):
#     if image is not None:
#         height, width, channels = image.shape
#         if width<height :
#             image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)                
#     preprocessed_image = cv2.resize(image, target_size)
#     reshaped_image = preprocessed_image.reshape(1, *preprocessed_image.shape)
#     image = reshaped_image.astype(np.float32)
#     image = image / 255.0
#     return image

# def predict_prawn(img):
#     print('White Prawn Model!!')
#     target_size=(280, 180)
#     label_map = {'Bad': 0, 'Good': 1}
#     preprocessed_img = preprocess_img(img, target_size)
#     prediction = model.predict(preprocessed_img)
#     prediction = np.array([np.argmax(pred) for pred in prediction])  
#     reverse_label_map = {idx: img_type for img_type, idx in label_map.items()}
#     predicted_class = reverse_label_map[prediction[0]] 
#     return predicted_class
