import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from keras.models import load_model
import base64
from ultralytics import YOLO
from . import Species_Classification, sardine, mackerel, prawns
import logging
yolo = YOLO(r'fish_product_models/best.pt')
prawn = YOLO(r'fish_product_models/prawn.pt')
cut = YOLO(r'fish_product_models/cut3.pt')

def extract_single_image_segment2(img, mask_segment, shape):
    height, width = shape[0], shape[1]
    segment = np.array(mask_segment, dtype=np.int32)
    segment = segment.reshape((-1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [segment], 255)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(segment)
    segment_image = np.zeros((h, w, 3), dtype=np.uint8)
    segment_image[0:h, 0:w] = masked_img[y:y+h, x:x+w]
    black_mask = np.all(segment_image == [0, 0, 0], axis=-1)
    segment_image[black_mask] = [255, 255, 255]
    return segment_image

def extract_single_image_segment(img, mask_segment, shape, background='black'):
    height, width = shape[0], shape[1]
    segment = np.array(mask_segment, dtype=np.int32)
    segment = segment.reshape((-1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [segment], 255)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(segment)
    segment_image = np.zeros((h, w, 3), dtype=np.uint8)
    segment_image[0:h, 0:w] = masked_img[y:y+h, x:x+w]

    if background.lower() == 'black':
        black_mask = np.all(segment_image == [0, 0, 0], axis=-1)
        segment_image[black_mask] = [0, 0, 0]  # Set background to black
    elif background.lower() == 'white':
        black_mask = np.all(segment_image == [0, 0, 0], axis=-1)
        segment_image[black_mask] = [255, 255, 255]  # Set background to white
    else:
        raise ValueError("Invalid background color. Use 'black' or 'white'.")

    return segment_image

def process_image2(image, size=(640, 640)): # for eye model
    h, w = image.shape[:2]
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = size[0]
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = size[1]
        new_w = int(new_h * aspect_ratio)

    resized_image = cv2.resize(image, (new_w, new_h))

    pad_h = (size[1] - new_h) // 2
    pad_w = (size[0] - new_w) // 2
    padded_image = cv2.copyMakeBorder(
        resized_image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    padded_image = cv2.resize(padded_image, (640,640))
    return np.asarray(padded_image)


def is_cut(image):
    img = image[:,:,::-1]
    res = cut.predict(img, conf=0.75)
    if len(res[0].boxes.xyxy) == 0:
        return False
    else:
        return True

def return_mask(image, masks, labels, boxes):
    mask_image = np.zeros_like(image)

    height, width, _ = image.shape

    for mask, label, box in zip(masks, labels, boxes):
        xmin, ymin, xmax, ymax = list(map(int, box))

        # Calculate the thickness and text size based on image dimensions
        thickness = max(int(min(height, width) / 200), 1)
        text_size = max(int(min(height, width) / 800), 1)

        if 'good' in label.lower():
            color = (0, 255, 0)  # Green mask
        elif 'ok' in label.lower():
            color = (255, 255, 0)  # Yellow mask
        else:
            color = (255, 0, 0)  # Red mask

        mask_coords = np.array(mask, dtype=np.int32)
        cv2.fillPoly(mask_image, [mask_coords], color)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, text_size, color, thickness)

    alpha = 0.2  # Transparency level for the mask
    image_with_masks = cv2.addWeighted(image, 1 - alpha, mask_image, alpha, 0)
    image_with_masks = cv2.cvtColor(image_with_masks, cv2.COLOR_BGR2RGB)
    # image_with_masks = cv2.resize(image_with_masks, (512, 512))
    return image_with_masks

def freshness_prediction(image, w_mask_image, b_mask_image, fish):
    if fish.lower() == 'sardine':
      damage = is_cut(process_image2(w_mask_image))
      if damage:
         return 'Bad','Damaged'
      else:
         pred = sardine.predict_sardine(b_mask_image)
         if pred == 'Bad':
            return pred, 'Softness'
         else:
            return pred, None
    elif fish.lower() == 'mackerel':
      damage = is_cut(process_image2(w_mask_image))
      if damage:
         return 'Bad', 'Damaged'
      else:
         pred = mackerel.predict_mackerel(b_mask_image)
         if pred == 'Bad':
            return pred, 'Softness'
         else:
            return pred, None
      
    elif fish.lower().replace(" ", "") == 'whiteprawn':
      pred = prawns.predict_prawn(b_mask_image)
      if pred == 'Bad':
            return pred, 'Softness'
      else:
         return pred, None

    else:
      damage = is_cut(process_image2(w_mask_image))
      if damage:
         return 'Bad', 'Damaged'
      else:
         return 'Good', None


def final_prediction(image, species, model):
  result = dict()
  top3 = []
  image = Image.open(image)
  image = np.asarray(image)
  shape = image.shape
  try:
      if model.lower() == 'fish':
         pred = yolo.predict(image[:,:,::-1], conf=0.75)
      else:
         pred = prawn.predict(image[:,:,::-1], conf=0.5)
  except:
     return None

  if len(pred[0].boxes.xyxy) == 0 or pred[0].masks == None:
     return None
  boxes = pred[0].boxes.xyxy
  masks=pred[0].masks.xy
  filtered_boxes = sorted(boxes, key=lambda bbox: bbox[0])
  sorted_masks = [masks[i] for i in sorted(range(len(masks)), key=lambda x: boxes[x][0])]
#   xmin, ymin, xmax, ymax = list(map(int,filtered_boxes[0]))
#   extracted_image = image[ymin:ymax, xmin:xmax]
#   species_pred = str(Species_Classification.test_img(image))
#   if species_pred.lower().replace(" ", "") == species.lower().replace(" ", ""):
#      feedback = 'True'
#      logging.info('WRONG CLASSIFICATION!!!!')
#   else:
#      feedback='False'
#      logging.info('CORRECT CLASSIFICATION!!!!')
  new_labels=[]
  reasons = []
  i=0
  for box, mask in zip(filtered_boxes, sorted_masks):
    i+=1
    xmin, ymin, xmax, ymax = list(map(int,box))
    extracted_image = image[ymin:ymax, xmin:xmax]
    w_mask_image = extract_single_image_segment(image, mask, shape = shape, background='white')
    b_mask_image = extract_single_image_segment(image, mask, shape = shape, background='black')
    L,R = freshness_prediction(extracted_image, w_mask_image, b_mask_image, species)
    v = str(i)+'-'+str(L)
    if R == None:
       v = str(i)+'-'+str(L)
    else:
       v = str(i)+'-'+str(L)+'-'+str(R)
    reasons.append(R)
    new_labels.append(v)
  output_image = return_mask(image, sorted_masks, new_labels,filtered_boxes)
  _, image_bytes = cv2.imencode('.jpg', output_image)
  encoded_string = base64.b64encode(image_bytes).decode('utf-8')
  print("Number of fishes in the image: ",len(filtered_boxes))
  sorted_list = sorted(new_labels, key=lambda x: int(x.split('-')[0]))
  good = 0
  bad = 0
  for i in range(len(sorted_list)):
    fresh = sorted_list[i].split('-')[1]
    top3.append(fresh)
    if fresh.lower() == 'good':
      good+=1
    else:
      bad+=1
  result['fishes-detected'] = good + bad
  result['good-fishes'] = good
  result['bad-fishes'] = bad
  result['reasons'] = reasons
  result['image-encode'] = encoded_string
  result['classification'] = 'True'
  result['species'] = species
  result['first3'] = top3
  return result
