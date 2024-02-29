from django.shortcuts import render
from django.http import HttpResponse
from django.http import FileResponse
from django.core.files.storage import FileSystemStorage, default_storage
from rest_framework.response import Response
from django.core.files.base import ContentFile 
import requests
from django.http import JsonResponse
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
import pickle
import io, os
# from PIL import Image, ImageDraw
import PIL
import cv2
import logging
import psycopg2
import sys
import pytz
from datetime import datetime
# import glob
from .models import Results, Feedback
import base64
from io import BytesIO
from PIL import Image
import numpy as np
# import torch
# import torch.nn as nn
# from torchvision import datasets, models, transforms

# deep learning libraries
import tensorflow as tf
import keras
from tensorflow.keras.utils import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image
# from sklearn.decomposition import PCA
# from sklearn import svm
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.views import APIView
import boto3
from .updated_output import final_prediction
from .tasks import run_upload_to_s3, run_create_result_with_retry, get_latest_user
####################################################
####################################################
####################################################

IMG_SIDE = 360

Log_Format = "%(levelname)s %(asctime)s - %(message)s"

abs_path = ""

handler = logging.handlers.RotatingFileHandler(
    filename=f"{abs_path}logfile.log",
    mode='a',
    maxBytes=1024*1024,
    backupCount=3,
    encoding=None,
    delay=10
)

logging.basicConfig(
                    # filemode = "a",
                    format = Log_Format, 
                    level = logging.INFO,
                    handlers=[handler])


##########################################################################################

# color global constants
RED = {'R':255,'G':17,'B':0}
BANANA_YELLOW = {'R':254,'G':221,'B':0}
DARK_YELLOW = {'R':173,'G':156,'B':2}
GREEN = {'R':0,'G':169,'B':6}
DARK_GREEN = {'R':0,'G':102,'B':0}

def banana_freshness_index(val):
    return val + 1

# input will range from 1 to 7, all intergers
def banana_color_code(val):
    if val >= 6:
        return DARK_YELLOW
    elif (val <= 5 and val >= 4):
        return BANANA_YELLOW
    else:
        return DARK_GREEN

def banana_image_preprocessing(img_np):
    # img=cv2.resize(img_np,(256,256))
    # img = img / 255.
    img_np = img_np.resize((256,256))
    img_np = img_to_array(img_np)
    img_np = img_np / 255.
    
    #print(img)
    # logging.info(f'img after preprocessing = {img_np}')
    return img_np

def get_banana_prediction(processed_img):
    # logging.info('get_banana_prediction entered')
    # preds= COMMODITY_MAP['BANANA']['model'].predict(tf.expand_dims(processed_img,axis=0))
    preds = COMMODITY_MAP['BANANA']['model'].predict(processed_img.reshape((1, processed_img.shape[0], processed_img.shape[1], processed_img.shape[2])))
    prediction = preds.argmax(axis=1)[0]
    # logging.info('get_banana_prediction about to return')
    return prediction

def freshnessIndex(val):
    return val

def fishColorCode(val):
    if val > 70:
        return GREEN
    elif (val < 70 and val > 30):
        return DARK_YELLOW
    else:
        return RED

def fish_image_preprocessing(img_np):
    img_size = img_np.size[0]

    crops = []
    crop_part = ['center','top','bottom','left','right']
    # Defining the axes of for cropping the image into 5 parts
    cord1 = 0
    cord2 = int(img_size/3)
    cord3 = int(2*img_size/3)
    cord4 = img_size

    # # Getting the crops of the image [cv2 version]
    # center = img_np[cord2:cord3, cord2:cord3] #center
    # top = img_np[cord1:cord2, cord2:cord3] #top
    # bottom = img_np[cord3:cord4, cord2:cord3] #bottom
    # left = img_np[cord2:cord3, cord1:cord2] #left
    # right = img_np[cord2:cord3, cord3:cord4] #right

    # # Getting the crops of the image [PIL version]
    center = img_np.crop((cord2, cord2, cord3, cord3)) #center
    top = img_np.crop((cord2, cord1, cord3, cord2)) #top
    bottom = img_np.crop((cord2, cord3, cord3, cord4)) #bottom
    left = img_np.crop((cord1, cord2, cord2, cord3)) #left
    right = img_np.crop((cord3, cord2, cord4, cord3)) #right

    # Storing all the crops in a list
    crops.append(center)
    crops.append(top)
    crops.append(bottom)
    crops.append(left)
    crops.append(right)

    # Iterating through the cropped image parts
    for i in crop_part:
        image_crp = crops[crop_part.index(i)]
        crop_partx = i

        # Un-comment the next lines to display the cropped parts of the image
        # print("Before_crop: ",image_crp.shape)
        # cv2.imshow(f"Face_{crop_partx}", image_crp)
        # key_press = cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # cv2 
        # # Checking and resizing the cropped images to 100X100
        # if ((image_crp.shape)[0] != 100 or (image_crp.shape)[1] != 100):
        #     # print('Resizing image...')
        #     imResize = cv2.resize(image_crp,(100,100))
        #     # print("After_crop: ",imResize.shape)
        #     crops[crop_part.index(i)] = imResize
        #     # print(type(crops[crop_part.index(i)]))
        #     # print(crops[crop_part.index(i)].shape)
        #     # cv2.imshow(f"Face_{crop_partx}", crops[crop_part.index(i)])
        #     # key_press = cv2.waitKey(0)
        #     # cv2.destroyAllWindows()

        #     # normalize it
        #     crops[crop_part.index(i)] = crops[crop_part.index(i)] / 255.

        # PIL 
        # Checking and resizing the cropped images to 100X100
        if ((image_crp.size)[0] != 100 or (image_crp.size)[1] != 100):
            # print('Resizing image...')
            imResize = image_crp.resize((100,100))
            # print("After_crop: ",imResize.size)
            
            # Converting the PIL object to array
            imResize = img_to_array(imResize)
            # normalize it
            imResize = imResize / 255.
            
            crops[crop_part.index(i)] = imResize

    return crops

    # img=cv2.resize(img_np,(100,100))
    # #print(img)
    # img = img/255.
    # return img

# fish_num_preprocess = transforms.Compose([
#     transforms.Resize(900),
#     transforms.ToTensor(),
#     utils.normalize_transform()
# ])

def get_fish_prediction(processed_imgs):
    final_prediction_percentage = 0
    for i in range(len(processed_imgs)):
        # preds will be 0 or 1
        # preds = COMMODITY_MAP['FISH']['model'].predict(tf.expand_dims(processed_imgs[i],axis=0))
        preds = COMMODITY_MAP['FISH']['model'].predict(processed_imgs[i].reshape((1, processed_imgs[i].shape[0], processed_imgs[i].shape[1], processed_imgs[i].shape[2])))
        logging.info(f"preds = {preds}")
        final_prediction_percentage += (round(preds.ravel()[0]) * 20)
    # temporary fix [As model gives opposite prediction result]
    final_prediction_percentage = 100 - final_prediction_percentage
    return final_prediction_percentage

# def get_fish_num_prediction(model, torch_img):
#     # image = Image.open(image_file)
#     # img = fish_num_preprocess(torch_image)
#     threshold = 90
#     predictions = model.predict(torch_img)
#     labels, boxes, scores = predictions
#     thresh=threshold/100
#     filtered_indices=np.where(scores>thresh)
#     filtered_scores=scores[filtered_indices]
#     class_idx = len(filtered_scores)
#     return class_idx
 
##########################################################################################
# Tensorflow MODEL LOADING
##########################################################################################

def get_loaded_model(model_filename):
    #load model
    loaded_model=load_model(model_filename)
    logging.info(f'Successfully loaded model from file : {model_filename}')
    return loaded_model


# def generate_mod(model_filename):
#     # res_mod = models.resnet18(pretrained=True)
#     # num_ftrs = res_mod.fc.in_features
#     # print(num_ftrs)
#     # # res_mod.fc = nn.Linear(num_ftrs, 6)
#     # # res_mod.classifier[1] = nn.Linear(res_mod.last_channel, 9)
#     # # res_mod.classifier[1] = torch.nn.Linear(in_features=res_mod.classifier[1].in_features, out_features=10)
#     # for param in res_mod.parameters():
#     #     param.requires_grad = False
#     # fc_inputs = res_mod.fc.in_features

#     # res_mod.aux_logits=False
#     # res_mod.trainable = True

#     # res_mod.fc = nn.Sequential(
#     #                             nn.Linear(fc_inputs, num_ftrs),
#     #                             nn.ReLU(),
#     #                             nn.Dropout(0.4),
#     #                             nn.Linear(num_ftrs, 7),
#     #                             nn.LogSoftmax(dim=1)# For using NLLLoss()
#     #             )

#     # res_mod.load_state_dict(torch.load(model_filename,map_location=torch.device('cpu')))
#     logging.info(f'MODEL from file - {model_filename} LOADED SUCCESSFULLY!')
#     return 

# banana model file name should be BANANA_model.sav
# fish model file name should be FISH_model.sav

COMMODITY_MAP = {
    'BANANA':{
        'prefix':'Stage',
        'num_classes':8,
        'result':banana_freshness_index,
        'colorCode':banana_color_code,
        'image_preprocessing': banana_image_preprocessing,
        'predict': get_banana_prediction
        },
    'FISH':{
        'prefix':'Freshness Index',
        'num_classes':7,
        'result':freshnessIndex,
        'colorCode':fishColorCode,
        'image_preprocessing': fish_image_preprocessing,
        'predict': get_fish_prediction
        },
    # 'FISH_NUM':{
    #     'image_preprocessing': fish_num_preprocess,
    #     'predict': get_fish_num_prediction
    # },
}

# for commodity_key in COMMODITY_MAP:
#     COMMODITY_MAP[commodity_key]['model'] = generate_mod(f"models/{commodity_key}_model.sav")
#     logging.info(f"{commodity_key}'s model = {COMMODITY_MAP[commodity_key]['model']}")

#COMMODITY_MAP['BANANA']['model'] = get_loaded_model(r'/home/ec2-user/djangotest/models/convosrcimgsetval93train89.h5')
#COMMODITY_MAP['FISH']['model'] = get_loaded_model(r'/home/ec2-user/djangotest/models/fishmodel.h5')
# COMMODITY_MAP['FISH_NUM']['model'] = core.Model.load(r'C:\Users\hp\Desktop\MobileAPI\djangotest\models\model_weights_improved.pth', ['Fish'])



##########################################################################################
# RESNET MODEL LOADING
##########################################################################################

##########################################################################################
# SVM PCA MODEL LOADING
##########################################################################################

# import pickle
# # filename = 'Fish_SVM_model_cropped_data.sav'
# filename = 'pipe_model_PCA50_SVM_Fish.sav'
# SVM_PCA_MODEL = pickle.load(open(filename, 'rb'))

##########################################################################################
# SVM PCA MODEL LOADING
##########################################################################################

# def normalize(img_pil):
#     # img_pil.save('tmp/norm_in.png')
#     h,w = img_pil.size
#     # creating luminous image
#     lum_img = Image.new('L',[h,w] ,0)
#     draw = ImageDraw.Draw(lum_img)
#     draw.pieslice([(50,50),(h-50,w-50)],0,360,fill=1)
#     img_arr = np.array(img_pil)
#     lum_img_arr = np.array(lum_img)
#     tool  = Image.fromarray(lum_img_arr)
#     # display(Image.fromarray(lum_img_arr))
#     tool_img = np.array(tool.convert('RGB'))
#     # print('tool',np.array(tool_img))
#     final_img_arr = np.dstack((img_arr, lum_img_arr))
#     # display(Image.fromarray(final_img_arr))
#     cropped = Image.fromarray(final_img_arr)
#     # display(cropped)
#     final_img = cropped.convert('RGB')
#     # print('fin?',final_img)
#     # final_img.save('tmp/norm.png')
#     return Image.fromarray(np.array(final_img*tool_img))

#     ######################################
#     #### THIS IS OLD FUNCTION 
#     # img_pil.save('tmp/norm_in.png')
#     h,w = img_pil.size
#     # creating luminous image
#     lum_img = Image.new('L',[h,w] ,0)
#     draw = ImageDraw.Draw(lum_img)
#     draw.pieslice([(50,50),(h-50,w-50)],0,360,fill=255)
#     img_arr = np.array(img_pil)
#     lum_img_arr = np.array(lum_img)
#     tool  = Image.fromarray(lum_img_arr)
#     # display(Image.fromarray(lum_img_arr))
#     tool_img = np.array(tool.convert('RGB'))
#     # print('tool',np.array(tool_img))
#     final_img_arr = np.dstack((img_arr, lum_img_arr))
#     # display(Image.fromarray(final_img_arr))
#     cropped = Image.fromarray(final_img_arr)
#     # display(cropped)
#     final_img = cropped.convert('RGB')
#     # print('fin?',final_img)
#     ret = np.array(final_img*tool_img)
#     ###########################################
#     # SAVE NORMALIZED IMAGE
#     img_save = Image.fromarray(ret)
#     # img_save.save('tmp/norm.png')
#     ###########################################

#     return ret
#     #### THIS IS OLD FUNCTION
#     ######################################

# mean_nums = [0.485, 0.456, 0.406]
# std_nums = [0.229, 0.224, 0.225]
# chosen_transforms = { 'val': transforms.Compose([
#         transforms.Resize((360,360)),
# #         transforms.CenterCrop(224),
#         # transforms.ColorJitter( contrast= [1,1],saturation=[3,3]),
#         transforms.Lambda(normalize),
#         transforms.ToTensor(),
# #         transforms.Grayscale(),
#         # transforms.Normalize(mean_nums, std_nums),
# ]),
# }

# test_mod = res_mod
# num_ftrs = test_mod.fc.in_features
# test_mod.fc = nn.Linear(num_ftrs, 9)
# test_mod.load_state_dict(torch.load('fish_init.sav',map_location=torch.device('cpu')))

# for file_name in glob.glob('djangotestapp/test/*.png'):
#     test_img = Image.open(file_name)
#     test_outs = test_mod(torch.unsqueeze(chosen_transforms['val'](test_img), 0))
#     _,test_pred = torch.max(test_outs,1)
#     logging.info(f'IMAGE : {file_name}, TEST PREDS : {test_pred}')

def test(request):
    return HttpResponse("This is test api")

def sendData(request):
    return JsonResponse({'foo': 'sendDataDone'})

def saveStreamedImage(data):
    logging.info(f'TYPE data read : {type(data.read())}')
    return default_storage.save('tmp/curr_cap.jpg', ContentFile(data.read()))

def cropImage(path):
    image = cv2.imread(path)
    verticalCenter,horizontalCenter = image.shape[0]/2,image.shape[1]/2
    x1,y1 = int(horizontalCenter-140),int(verticalCenter-140)
    x2,y2 = int(horizontalCenter+140),int(verticalCenter+140)
    # cv2.rectangle(image,(x1,y1),(x2,y2),(255,255,0),2) #Draw rectangle for crop reference
    image = image[ y1:y2 , x1:x2 ] # Cropping image
    cv2.imwrite(path,image) #Replacing the original image with cropped image

def fetch_last_img_id(con):
    try:
        # Defining fetch query
        fetch_query = 'SELECT imgd_id FROM public."imageData_imagedata" ORDER BY imgd_id desc LIMIT 1'

        cur = con.cursor()
        cur.execute(fetch_query)
        
        # Retrieving the bytearray image data
        query_res = cur.fetchall()
    #     con.commit()
        # logging.info(type(query_res))

    except psycopg2.DatabaseError as e:
        if con:
            con.rollback()
        logging.error('Error %s' % e) 
        sys.exit(1)

    finally:
        # if con:
        #     con.close()
        return query_res[0][0] + 1

def dbPush(brand,model,data,con,prediction,remarks,part,hour,orig,hw,flash):
    try:
    # Ignore the var nullx
        # nullx = 'PREV COLUMN'

        # now = datetime.now()
        # date_stamp = str(now.date())
        # time_stamp = str(now.time())

        cur_date_time = datetime.now(pytz.timezone("Asia/Calcutta"))
        date_stamp = str(cur_date_time).partition(" ")[0]
        time_stamp = str(cur_date_time).partition(" ")[2].partition("+")[0]

        s3_key = str(cur_date_time).split('+')[0].replace(' ','_').replace(':','-').replace('.','-') + ".jpg"
        # hour = hour.split('_')[1]

        s3 = boto3.client('s3')
        
        # imgd_id needs to be unique in the DB
        # imgd_id = img_id
        imgd_device_make = brand
        imgd_device_model = model
        
        # Insert Query
        insert_query = """INSERT INTO public."imageData_imagedata"
                        (imgd_date,imgd_timestamp,imgd_device_make,imgd_device_model,
                        imgd_binimg,imgd_pred,imgd_remarks,imgd_body_part,imgd_commodity,
                        imgd_true_label,imgd_s3_key,imgd_binimg_orig,imgd_height_width,flash)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        
        # Parameters for the INSERT query
        params = (date_stamp,time_stamp,imgd_device_make,imgd_device_model,data,prediction,remarks,part,'FISH',hour,s3_key,orig,hw,flash) #converted_img -> binary

        cur = con.cursor()
        cur.execute(insert_query, params)
        con.commit()

    except psycopg2.DatabaseError as e:
        if con:
            con.rollback()
        logging.error('Error %s' % e) 
        sys.exit(1)

    finally:
        if con:
            con.close()

# def readImage():
#     try:
#         fin = open('tmp/curr_cap.jpg', "rb")
#         converted_img = bytearray(fin.read())
#         logging.info(type(converted_img))
#         return converted_img

#     except IOError as e:
#         logging.info("Error %d: %s" % (e.args[0],e.args[1]))
#         sys.exit(1)

#     finally:
#         if fin:
#             fin.close()

# def pieslice_img(img_pil):
#     img = img_pil
#     h,w = img.size
#     # creating luminous image
#     lum_img = Image.new('L',[h,w] ,1)
#     draw = ImageDraw.Draw(lum_img)
#     draw.pieslice([(50,50),(h-50,w-50)],0,360,fill=255)
#     img_arr = np.array(img)
#     lum_img_arr = np.array(lum_img)
# #     display(Image.fromarray(lum_img_arr))
#     final_img_arr = np.dstack((img_arr, lum_img_arr))
# #     display(Image.fromarray(final_img_arr))
#     return Image.fromarray(final_img_arr)

# def crop_pil(img):
#     width, height = img.size

#     # TO STORE WIDTH AND HEIGHT of original image TO DB
#     wh_db = str(width) + ',' + str(height)
     
#     # GETS AN IMAGE OF H,W consuming the maximum width or height
#     IMG_SIDE = width if width < height else height
    
#     cx = int(width / 2)
#     cy = int(height / 2)
 
#     ######################################################
#     # FOR BOTTOM CROP, FOR FLUTTER LISTVIEW - REVERSE = TRUE, ASPECT RATIO 1 :D
#     left = width - height
#     right = width
#     upper = 0
#     bottom = height
#     ######################################################

#     ######################################################
#     # CROP THE CENTER SQUARE OF THE IMAGE, ASPECT RATIO SUCKED ON FLUTTER :(
#     # left= cx - (IMG_SIDE/2)
#     # upper = cy - (IMG_SIDE/2)
#     # bottom = upper + IMG_SIDE
#     # right = left + IMG_SIDE
#     ######################################################

#     logging.info(f'CX : {cx}, CY : {cy},')
#     logging.info(f'LEFT : {left}, RIGHT : {right}, UPPER : {upper}, BOTTOM : {bottom}')
#     # return img.crop((left,upper,right,bottom)).rotate(-90), img.rotate(-90,expand=True),wh_db
#     return img.crop((left,upper,right,bottom)), img,wh_db

# def toImgOpenCV(imgPIL): # Conver imgPIL to imgOpenCV
#     i = np.array(imgPIL) # After mapping from PIL to numpy : [R,G,B,A]
#                          # numpy Image Channel system: [B,G,R,A]
#     red = i[:,:,0].copy(); i[:,:,0] = i[:,:,2].copy(); i[:,:,2] = red
#     return i; 


# TENSOR_TO_PIL = transforms.ToPILImage()

#like on_message
# @csrf_exempt
           #logs
                 #   bad = result['bad-fishes']
                  #  good = result['good-fishes']
                   # total = bad+good
                    #image = result['image-encode']
                    #return Response({'Fishes detected': total, 'Good fishes': good, 'bad fishes': bad, 'image': image}, status=status.HTTP_200_OK)
                #else:
                    #return Response({'message': 'Bad image'}, status=status.HTTP_403_FORBIDDEN)
                #fishname = fields['FishName']
                #freshness = freshness_prediction(request.data['capture'], fishname)

                #return Response({'Freshness':freshness}, status=status.HTTP_200_OK)
                # return Response({'Count':count,'test':test}, status=status.HTTP_200_OK)
        # except Exception as e:
        #     logging.error(f"Error from postdata {e}")
        #     return JsonResponse({'result': 'Error','numericVal':-1})
        # finally:
        #     # Camtest is used for testing and ignore dbpush
        #     # isTest variable is assigened True if its testing from mobile or on API
        #     isTest = True if test == 'True' or hour == 'CAM_TEST'  else False
        #     remarks = 'TEST_TEST' if isTest else 'CAPTAIN_FRESH'
        #     logging.info(f'TEST VAR : {isTest}, type : {type(isTest)}')
        #     if isTest == False:
        #         try:
        #             logging.info('DB PUSH STARTED!')
        #             con = psycopg2.connect(database = 'ebdb',
        #                                     user = 'qZenseTEST',
        #                                     password = 'eFirst2019',
        #                                     host = 'dev-rdstest.cbudtrkx2byn.ap-south-1.rds.amazonaws.com',
        #                                     port =  5432)
        #             logging.info('CONN EST!')
        #             # dbPush(brand,deviceModel,image_cropped.tobytes(),con,str(res_numeric),'CAPTAIN_FRESH',part,hour,image.tobytes(),wh,str(flash))
        #             dbPush(brand,deviceModel,TENSOR_TO_PIL(transformed_image).tobytes(),con,str(res_numeric),remarks,part,hour,img_np.tobytes(),wh,str(flash))
        #             logging.info('Image Pushed Succesfully!')
        #             if con:
        #                 con.close()
        #             logging.info('Connection closed!')
        #         except Exception as e:
        #             logging.info('Error in DB section!')
        #             logging.info(e)
        #             if con:
        #                 con.close()
        #             logging.info('Connection closed!')
        #     else:
        #         logging.info(f'NOT PUSHING TO DB because TEST is {isTest}')
        #     logging.info('DONE!')



result = {}
input_url=""
result_url=""
top3 = []

class postDataView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    def post(self, request):
        # try:
        if request.method=="POST":
            logging.info('RECIEVED a POST REQUEST!')
            logging.info('VIEWS DL RUNNING!')
            fields = request.POST
            deviceModel,brand,test = fields["deviceModel"],fields["brand"],fields["test"]

            mlModel = fields['mlModel']
            part = fields['part']
            hour = fields['hour']
            flash = fields['flash']

            #logs
            logging.info(f'Device model : {deviceModel}')
            logging.info(f'Device Brand : {brand}')
            logging.info(f'ML model : {mlModel}')
            logging.info(f'Part : {part}')
            logging.info(f'Hour : {hour}')
            logging.info(f'TEST FIELD FROM DEVICE : {test}, type : {type(test)}')

            if part.lower() == "gills" or mlModel.lower() == 'banana':
                # COMMODITY_MAP['BANANA']['model'] = get_loaded_model(r'C:\Users\KARTIK\Desktop\Mobile_APP\models\convosrcimgsetval93train89.h5')
                # COMMODITY_MAP['FISH']['model'] = get_loaded_model(r'C:\Users\KARTIK\Desktop\Mobile_APP\models\fishmodel.h5')
                try:  
                    COMMODITY_MAP['BANANA']['model'] = get_loaded_model(r'/home/ec2-user/ML/models/convosrcimgsetval93train89.h5')
                    COMMODITY_MAP['FISH']['model'] = get_loaded_model(r'/home/ec2-user/ML/models/fishmodel.h5')
                except:  
                    COMMODITY_MAP['BANANA']['model'] = get_loaded_model(r'models/convosrcimgsetval93train89.h5')
                    COMMODITY_MAP['FISH']['model'] = get_loaded_model(r'models/fishmodel.h5')
                #select model
                curr_model = COMMODITY_MAP[mlModel]
                #Bytes to PIL convert
                imgBytes = request.FILES['capture'].read()

                image = PIL.Image.open(io.BytesIO(imgBytes)) # Whole image PIL CLASS
                # if mlModel == 'FISH':
                #     num_img = COMMODITY_MAP['FISH_NUM']['image_preprocessing'](image)
                #     FishNum_model = COMMODITY_MAP['FISH_NUM']['model']
                #     num = COMMODITY_MAP['FISH_NUM']['predict'](FishNum_model, num_img)
                    # print(num)
                logging.info(f'received shape = {image.size[0]}, {image.size[1]}')

                #  image preprocessing
                img = curr_model['image_preprocessing'](image)

                logging.info('Image preprocessing done')

                # image prediction
                prediction = curr_model['predict'](img)
                logging.info(f'Prediction for the current image : {prediction}')

                prefix = curr_model['prefix']
                logging.info(f'prediction : {prediction}')
                res_numeric = curr_model["result"](prediction)
                color_code = curr_model['colorCode']
                responseJson = {'result': f'{prefix} {res_numeric}','numericVal':int(res_numeric)}
                responseJson.update(color_code(int(res_numeric)))
                # if mlModel == 'FISH':
                #     responseJson['NumberOfFishes'] = num
                # Make error here to test
                ###############################
                # VARIABLE HELLO DOES NOT EXIT, UNCOMMENT THIS LINE TO CREATE ERROR IN CODE
                ###############################
                logging.info(f'RESPONSE Json : {responseJson}')
                return JsonResponse(responseJson)
            else:
                fishname = fields['FishName']
                capture_image = request.data['capture']
                result = final_prediction(capture_image, fishname, mlModel)
                if result == None:
                    return Response({'message': 'Fish not detected!'}, status=status.HTTP_403_FORBIDDEN)
                bad = result['bad-fishes']
                good = result['good-fishes']
                total = bad+good
                reasons = reasons = result['reasons']
                # AWS credentials
                # AWS_ACCESS_KEY_ID = 'AKIA2QMGQOEAXYO3BIYV'
                # AWS_SECRET_ACCESS_KEY = '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'
                AWS_STORAGE_BUCKET_NAME = 'mobile-api-results'
                AWS_S3_CUSTOM_DOMAIN = 'mobile-api-results.s3.ap-south-1.amazonaws.com'
                current_date = datetime.now().strftime('%Y-%m-%d')
                current_timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d_%H:%M:%S')
                # Save input image to S3 and get the S3 URL
                # s3 = boto3.client(
                #     's3',
                #    aws_access_key_id=AWS_ACCESS_KEY_ID,
                #     aws_secret_access_key=AWS_SECRET_ACCESS_KEY
                # )
                last_result = Results.objects.order_by('-serial_number').first()
                last_serial_number = last_result.serial_number
                serial_number = last_serial_number + 1  # Get the next serial number
                input_image_name = f'{current_timestamp}_({serial_number})_{fishname}_input.jpeg'
                capture_image_bytes = capture_image.read()
                img = Image.open(capture_image)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format)
                img_byte_arr = img_byte_arr.getvalue()
                run_upload_to_s3(img_byte_arr ,AWS_STORAGE_BUCKET_NAME, f'inputs/{current_date}/{input_image_name}')
                input_image_url = f'https://{AWS_S3_CUSTOM_DOMAIN}/inputs/{current_date}/{input_image_name}'

                # Call freshness prediction and get the result image
                encoded_string = result['image-encode']
                decoded_bytes = base64.b64decode(encoded_string)
                image_array = np.frombuffer(decoded_bytes, np.uint8)
                freshness_result_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                result_image_bytes = cv2.imencode('.jpg', freshness_result_image)[1].tobytes()
                result_image_io = BytesIO(result_image_bytes)
                # Save result image to S3 and get the S3 URL
                result_image_name = f'{current_timestamp}_({serial_number})_{fishname}_result.jpeg'
                run_upload_to_s3(result_image_io, AWS_STORAGE_BUCKET_NAME, f'results/{current_date}/{result_image_name}')
                result_image_url = f'https://{AWS_S3_CUSTOM_DOMAIN}/results/{current_date}/{result_image_name}'
                run_create_result_with_retry(fishname, input_image_url, result_image_url, total, good, bad, reasons)
                # Create and save the entry in the new ImageData table
                # Results.objects.create(
                #     serial_number=serial_number,
                #     species_name=fishname,
                #     input_image_url=input_image_url,
                #     output_image_url=result_image_url,
                #     total_count=total,
                #     good_count = good,
                #     bad_count = bad,
                #     prediction_reason = reasons,
                #     user_name = get_latest_user(),
                # )
                if result:
                    bad = result['bad-fishes']
                    good = result['good-fishes']
                    total = bad+good
                    image = result['image-encode']
                    species = result['species']
                    feedback = result['classification']
                    reasons = result['reasons']
                    try:
                        with open(r'/home/ec2-user/DJANGOTEST/djangotestapp/result.pkl', 'wb') as file:
                            pickle.dump(result, file)
                    except:
                        with open(r'djangotestapp/result.pkl', 'wb') as file:
                            pickle.dump(result, file)
                    return Response({'Fishes detected': total,'Species':species, 'Species-Feedback':feedback, 'Good fishes': good, 'Bad fishes': bad, 'first3':result['first3'],'input_image_url':input_image_url, 'result_image_url':result_image_url, 'Image': image}, status=status.HTTP_200_OK)
                else:
                    return Response({'message': 'Fish not detected!'}, status=status.HTTP_403_FORBIDDEN)


class resultsFeedback(APIView):
    def post(self, request):
        if request.method == "POST":
            try:
                # Validate and retrieve data from the request
                f1_actual = request.data.get('f1_actual')
                f2_actual = request.data.get('f2_actual')
                f3_actual = request.data.get('f3_actual')
                f1_pred = request.data.get('f1_pred')
                f2_pred = request.data.get('f2_pred')
                f3_pred = request.data.get('f3_pred')
                input_image_url = request.data.get('input_image_url')
                result_image_url = request.data.get('result_image_url')
                if not all([input_image_url, result_image_url]):
                    return Response({'Message': 'Incomplete data provided'}, status=status.HTTP_400_BAD_REQUEST)

                with open('djangotestapp/result.pkl', 'rb') as file:
                    result = pickle.load(file)
                reasons = result['reasons']
                Feedback.objects.create(
                    input_image_url=input_image_url,
                    output_image_url=result_image_url,
                    fish1_actual=f1_actual,
                    fish2_actual=f2_actual,
                    fish3_actual=f3_actual,
                    fish1_pred=f1_pred,
                    fish2_pred=f2_pred,
                    fish3_pred=f3_pred,
                    prediction_reason = reasons,
                    result="Correct" if (f1_actual == f1_pred and f2_actual == f2_pred and f3_actual == f3_pred) else "False",
                    serial_number=Feedback.objects.count() + 1,
                    user_name = get_latest_user(),
                )

                return Response({'Message': 'Update Successful'}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({'Message': f'Update Failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
                
