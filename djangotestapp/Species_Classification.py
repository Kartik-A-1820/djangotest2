import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



fish_species = ['Mackerel', 'Seer', 'Sardine', 'Tuna', 
               'barracoda', 'basaa', 'emperor', 'Indian Salmon',
               'katala', 'ledi', 'papda', 'Pink Perch',
               'rohu', 'roopchand', 'Tilapia', 'Other']

def precision(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.reshape(y_true, (-1,))
    class_index = 0  # Class 3 index (0-based)
    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_index), 
                                                          tf.equal(y_pred, class_index)), 
                                           tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(tf.equal(y_pred, class_index), tf.float32))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())    
    return precision
def f1_score(y_true, y_pred, class_index=2):   
    tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_index), tf.equal(y_pred, class_index)), tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_index), tf.not_equal(y_pred, class_index)), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_true, class_index), tf.equal(y_pred, class_index)), tf.float32))
    precision = tp / (tp + fp) if tp > 0.0 else 0.0
    recall = tp / (tp + fn) if tp > 0.0 else 0.0
    if precision > 0.0 and recall > 0.0:
        f1_score = 2 * precision * recall / (precision + recall)  
    else :
        f1_score = 0.0
    return f1_score
def custom_accuracy(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.reshape(y_true, (-1,))
    l = np.arange(len(fish_species))
    f1_scores = []
    for class_index in l:
        score = f1_score(y_true, y_pred, class_index)
        f1_scores.append(score)
    macro_avg = sum(f1_scores)/len(f1_scores)
    return macro_avg
# classification_model = tf.keras.models.load_model(r'C:\Users\KARTIK\Desktop\Mobile_APP\models\Classification_model.h5',custom_objects={'custom_accuracy': custom_accuracy, 'precision':precision})
classification_model = None
num_classes = len(fish_species)
label_map = {species:i for i,species in enumerate(fish_species)}
priority_species = ['Mackerel', 'Seer', 'Sardine', 'Tuna']
priority_species_indices = []
for i, species in enumerate(priority_species):
    priority_species_indices.append(label_map[species])

def test_img(image):
    if not classification_model:
        #classification_model = tf.keras.models.load_model(r'C:\Users\KARTIK\Desktop\Mobile_APP\models\Classification_model.h5',custom_objects={'custom_accuracy': custom_accuracy, 'precision':precision})
        classification_model = tf.keras.models.load_model(r'/home/ec2-user/fish_product_models/Classification_model.h5',custom_objects={'custom_accuracy': custom_accuracy, 'precision':precision})
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    pred = classification_model.predict(image)
    pred = np.argmax(pred)
    if pred not in priority_species_indices:
        pred = 2
    return priority_species[pred]

