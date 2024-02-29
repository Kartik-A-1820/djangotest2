# from django.shortcuts import render
# from django.http import HttpResponse
# from django.http import FileResponse
# from django.core.files.storage import FileSystemStorage, default_storage
# from django.core.files.base import ContentFile 
# import requests
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt

# import io, os
# from PIL import Image, ImageDraw
# import cv2
# import logging
# import psycopg2
# import sys
# import pytz
# from datetime import datetime
# import glob

# import numpy as np

# import torch
# import torch.nn as nn
# from torchvision import datasets, models, transforms

# from sklearn.decomposition import PCA
# from sklearn import svm

# import boto3

# ####################################################
# ####################################################
# ####################################################

# IMG_SIDE = 360

# Log_Format = "%(levelname)s %(asctime)s - %(message)s"

# handler = logging.handlers.RotatingFileHandler(
#     filename="logfile.log",
#     mode='a+',
#     maxBytes=1024*1024,
#     backupCount=3,
#     encoding=None,
#     delay=10
# )

# logging.basicConfig(
#                     filemode = "w",
#                     format = Log_Format, 
#                     level = logging.INFO,
#                     handlers=[handler])


# ##########################################################################################
# # RESNET MODEL LOADING
# ##########################################################################################


# # res_mod = models.resnet18(pretrained=True)
# # num_ftrs = res_mod.fc.in_features
# # print(num_ftrs)
# # # res_mod.fc = nn.Linear(num_ftrs, 6)
# # # res_mod.classifier[1] = nn.Linear(res_mod.last_channel, 9)
# # # res_mod.classifier[1] = torch.nn.Linear(in_features=res_mod.classifier[1].in_features, out_features=10)
# # for param in res_mod.parameters():
# #     param.requires_grad = False
# # fc_inputs = res_mod.fc.in_features

# # res_mod.aux_logits=False
# # res_mod.trainable = True

# # res_mod.fc = nn.Sequential(
# #                             nn.Linear(fc_inputs, num_ftrs),
# #                             nn.ReLU(),
# #                             nn.Dropout(0.4),
# #                             nn.Linear(num_ftrs, 7),
# #                             nn.LogSoftmax(dim=1)# For using NLLLoss()
# #             )

# # res_mod.load_state_dict(torch.load('resnetmay5_6_aft_res18new_16_16.sav',map_location=torch.device('cpu')))
# # logging.info('MODEL LOADED SUCCESSFULLY!')

# ##########################################################################################
# # RESNET MODEL LOADING
# ##########################################################################################

# ##########################################################################################
# # SVM PCA MODEL LOADING
# ##########################################################################################

# import pickle
# # filename = 'Fish_SVM_model_cropped_data.sav'
# filename = 'pipe_model_PCA50_SVM_Fish.sav'
# SVM_PCA_MODEL = pickle.load(open(filename, 'rb'))

# ##########################################################################################
# # SVM PCA MODEL LOADING
# ##########################################################################################

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
#         # transforms.Lambda(normalize),
#         transforms.ToTensor(),
# #         transforms.Grayscale(),
#         # transforms.Normalize(mean_nums, std_nums),
# ]),
# }

# # test_mod = res_mod
# # num_ftrs = test_mod.fc.in_features
# # test_mod.fc = nn.Linear(num_ftrs, 9)
# # test_mod.load_state_dict(torch.load('fish_init.sav',map_location=torch.device('cpu')))

# # for file_name in glob.glob('djangotestapp/test/*.png'):
# #     test_img = Image.open(file_name)
# #     test_outs = test_mod(torch.unsqueeze(chosen_transforms['val'](test_img), 0))
# #     _,test_pred = torch.max(test_outs,1)
# #     logging.info(f'IMAGE : {file_name}, TEST PREDS : {test_pred}')

# def test(request):
#     return HttpResponse("This is test api")

# def sendData(request):
#     return JsonResponse({'foo': 'sendDataDone'})

# def saveStreamedImage(data):
#     logging.info(f'TYPE data read : {type(data.read())}')
#     return default_storage.save('tmp/curr_cap.jpg', ContentFile(data.read()))

# def cropImage(path):
#     image = cv2.imread(path)
#     verticalCenter,horizontalCenter = image.shape[0]/2,image.shape[1]/2
#     x1,y1 = int(horizontalCenter-140),int(verticalCenter-140)
#     x2,y2 = int(horizontalCenter+140),int(verticalCenter+140)
#     # cv2.rectangle(image,(x1,y1),(x2,y2),(255,255,0),2) #Draw rectangle for crop reference
#     image = image[ y1:y2 , x1:x2 ] # Cropping image
#     cv2.imwrite(path,image) #Replacing the original image with cropped image

# def fetch_last_img_id(con):
#     try:
#         # Defining fetch query
#         fetch_query = 'SELECT imgd_id FROM public."imageData_imagedata" ORDER BY imgd_id desc LIMIT 1'

#         cur = con.cursor()
#         cur.execute(fetch_query)
        
#         # Retrieving the bytearray image data
#         query_res = cur.fetchall()
#     #     con.commit()
#         # logging.info(type(query_res))

#     except psycopg2.DatabaseError as e:
#         if con:
#             con.rollback()
#         logging.error('Error %s' % e) 
#         sys.exit(1)

#     finally:
#         # if con:
#         #     con.close()
#         return query_res[0][0] + 1

# def dbPush(brand,model,data,con,prediction,remarks,part,hour,orig,hw,flash):
#     try:
#     # Ignore the var nullx
#         # nullx = 'PREV COLUMN'

#         # now = datetime.now()
#         # date_stamp = str(now.date())
#         # time_stamp = str(now.time())

#         cur_date_time = datetime.now(pytz.timezone("Asia/Calcutta"))
#         date_stamp = str(cur_date_time).partition(" ")[0]
#         time_stamp = str(cur_date_time).partition(" ")[2].partition("+")[0]

#         s3_key = str(cur_date_time).split('+')[0].replace(' ','_').replace(':','-').replace('.','-') + ".jpg"
#         # hour = hour.split('_')[1]

#         s3 = boto3.client('s3')
        
#         # imgd_id needs to be unique in the DB
#         # imgd_id = img_id
#         imgd_device_make = brand
#         imgd_device_model = model
        
#         # Insert Query
#         insert_query = """INSERT INTO public."imageData_imagedata"
#                         (imgd_date,imgd_timestamp,imgd_device_make,imgd_device_model,
#                         imgd_binimg,imgd_pred,imgd_remarks,imgd_body_part,imgd_commodity,
#                         imgd_true_label,imgd_s3_key,imgd_binimg_orig,imgd_height_width,flash)
#                         VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        
#         # Parameters for the INSERT query
#         params = (date_stamp,time_stamp,imgd_device_make,imgd_device_model,data,prediction,remarks,part,'FISH',hour,s3_key,orig,hw,flash) #converted_img -> binary

#         cur = con.cursor()
#         cur.execute(insert_query, params)
#         con.commit()

#     except psycopg2.DatabaseError as e:
#         if con:
#             con.rollback()
#         logging.error('Error %s' % e) 
#         sys.exit(1)

#     finally:
#         if con:
#             con.close()

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

# def freshnessIndex(val):
#     return round(-(val-5)*100/5,2)

# def fishColorCode(val):
#     if val > 70:
#         return GREEN
#     elif (val < 70 and val > 30):
#         return DARK_YELLOW
#     else:
#         return RED

# def toImgOpenCV(imgPIL): # Conver imgPIL to imgOpenCV
#     i = np.array(imgPIL) # After mapping from PIL to numpy : [R,G,B,A]
#                          # numpy Image Channel system: [B,G,R,A]
#     red = i[:,:,0].copy(); i[:,:,0] = i[:,:,2].copy(); i[:,:,2] = red
#     return i; 

# COMMODITY_MAP = {
#     'BANANA':{
#         'prefix':'Stage',
#         'model':'banana_init.sav',
#         'num_classes':8,
#         },
#     'FISH':{
#         'prefix':'Spoilage Index',
#         'model':'resnetmay55_eve_res18.sav',
#         'num_classes':7,
#         'result':freshnessIndex,
#         'colorCode':fishColorCode},
# }

# # color global constants
# RED = {'R':255,'G':17,'B':0}
# DARK_YELLOW = {'R':173,'G':156,'B':2}
# GREEN = {'R':0,'G':169,'B':6}

# TENSOR_TO_PIL = transforms.ToPILImage()

# #like on_message
# @csrf_exempt
# def postData(request):
#     try:
#         if request.method=="POST":
#             logging.info('RECIEVED a POST REQUEST!')
#             logging.info('VIEWS ML RUNNING!')
#             fields = request.POST
#             deviceModel,brand,test = fields["deviceModel"],fields["brand"],fields["test"]
#             mlModel = fields['mlModel']
#             part = fields['part']
#             hour = fields['hour']
#             flash = fields['flash']

#             #logs
#             logging.info(f'Device model : {deviceModel}')
#             logging.info(f'Device Brand : {brand}')
#             logging.info(f'ML model : {mlModel}')
#             logging.info(f'Part : {part}')
#             logging.info(f'Hour : {hour}')
#             logging.info(f'TEST FIELD FROM DEVICE : {test}, type : {type(test)}')
#             curr_model = COMMODITY_MAP[mlModel]
#             # res_mod.fc = nn.Sequential(
#             #                 nn.Linear(fc_inputs, num_ftrs),
#             #                 nn.ReLU(),
#             #                 nn.Dropout(0.4),
#             #                 nn.Linear(num_ftrs, curr_model['num_classes']),
#             #                 nn.LogSoftmax(dim=1)# For using NLLLoss()
#             # )
#             # res_mod.fc = nn.Linear(num_ftrs, curr_model['num_classes'])
#             # res_mod.load_state_dict(torch.load(curr_model['model'],map_location=torch.device('cpu')))
#             # logging.info('MODEL LOADED SUCCESSFULLY!')

#             #Bytes to PIL convert 
#             imgBytes = request.FILES['capture'].read()
#             logging.info('IMAGE READ FROM request fields')
#             # image = Image.open(io.BytesIO(imgBytes)) # Whole image PIL CLASS
#             image_cropped = Image.open(io.BytesIO(imgBytes)) # Whole image PIL CLASS
#             logging.info('PIL READ COMPLETE')
#             image_cropped,image,wh = crop_pil(image) # Cropped image PIL TYPE!    
#             logging.info(f'PIL SIZE : {image_cropped.size[0],image_cropped.size[1]}')  
#             wh = f'{image_cropped.size[0]},{image_cropped.size[1]}'     

#             ########################################
#             # PIESLICE AND RESIZE
#             # image_cropped.convert('RGB').save('tmp/curr_cap.jpg') # SAVE USING PIL
#             # image_cropped = pieslice_img(image_cropped)


#             image_cropped = image_cropped.resize((IMG_SIDE,IMG_SIDE))
#             logging.info(f'PIL SIZE AFTER RESIZE: {image_cropped.size[0],image_cropped.size[1]}') 
            
#             ########################################
#             # PIL TO CV2 conversion
#             # image_cropped.save('tmp/curr_cap_pil.png') # SAVE USING PIL
#             image_cropped = toImgOpenCV(image_cropped) # FORMAT IS CV2, till now it was PIL
#             # image_cropped = np.array(image_cropped.convert('RGB'))[:, :, ::-1].copy()
#             image_cropped = cv2.cvtColor(np.array(image_cropped), cv2.COLOR_RGB2BGR)
#             # logging.info(f'IMAGE SIZE : {image_cropped.shape}')
            
#             ########################################
            
#             # cv2.imwrite('tmp/curr_cap_cv2.png',image_cropped) # SAVE USING CV2
            
#             ########################################
#             logging.info('IMAGE CROPPED!')
#             # # image_cropped.save('tmp/curr_cap.png') # SAVE USING PIL
#             transformed_image = chosen_transforms['val'](Image.fromarray(image_cropped))
#             # # TENSOR_TO_PIL(transformed_image).save('tmp/curr_cap_trans.png')
#             # ############# DON"T MISS THIS FOR TRANSFER LEARNING!!!! #############
#             # inputs = torch.unsqueeze(transformed_image, 0)
#             # ############# DON"T MISS THIS FOR TRANSFER LEARNING!!!! #############
#             # res_mod.eval()
            
#             # outputs = res_mod(inputs)
#             # #Example prediction(outputs) ==> [0.1,0.2,0.3,0.4,0.6] is a list of probabilities for each class
#             # #Torch max to get the most likely class
#             # _,pred = torch.max(outputs,1)
#             ########################################
            
#             pred = SVM_PCA_MODEL.predict(image_cropped.flatten().reshape(1,388800))
            
#             logging.info('PREDICTION RUN!')
#             logging.info(f'OUTPUT : {pred}')
#             #####################
#             # RESNET
#             #####################
#             # pred_numpy = pred.numpy()[0]
#             #####################
#             # RESNET
#             #####################
#             pred_numpy = pred[0]
#             prefix = curr_model['prefix']
#             res_numeric = curr_model["result"](pred_numpy)
#             color_code = curr_model['colorCode']
#             responseJson = {'result': f'{prefix} {res_numeric}','numericVal':int(res_numeric)}
#             responseJson.update(color_code(int(res_numeric)))
#             # Make error here to test
#             ###############################
#             # VARIABLE HELLO DOES NOT EXIT, UNCOMMENT THIS LINE TO CREATE ERROR IN CODE
#             ###############################
#             logging.info(f'RESPONSE Json : {responseJson}')
#             return JsonResponse(responseJson)
#             # return JsonResponse({'result': 'Success'})
#     except Exception as e:
#         logging.error(e)
#         return JsonResponse({'result': 'Error','numericVal':-1})
#     finally:
#         # Camtest is used for testing and ignore dbpush
#         # isTest variable is assigened True if its testing from mobile or on API
#         isTest = True if test == 'True' or hour == 'CAM_TEST'  else False
#         remarks = 'TEST_TEST' if isTest else 'CAPTAIN_FRESH'
#         logging.info(f'TEST VAR : {isTest}, type : {type(isTest)}')
#         if isTest == False:
#             try:
#                 logging.info('DB PUSH STARTED!')
#                 con = psycopg2.connect(database = 'ebdb',
#                                         user = 'qZenseTEST',
#                                         password = 'eFirst2019',
#                                         host = 'dev-rdstest.cbudtrkx2byn.ap-south-1.rds.amazonaws.com',
#                                         port =  5432)
#                 logging.info('CONN EST!')
#                 # dbPush(brand,deviceModel,image_cropped.tobytes(),con,str(res_numeric),'CAPTAIN_FRESH',part,hour,image.tobytes(),wh,str(flash))
#                 dbPush(brand,deviceModel,TENSOR_TO_PIL(transformed_image).tobytes(),con,str(res_numeric),remarks,part,hour,image_cropped.tobytes(),wh,str(flash))
#                 logging.info('Image Pushed Succesfully!')
#                 if con:
#                     con.close()
#                 logging.info('Connection closed!')
#             except Exception as e:
#                 logging.info('Error in DB section!')
#                 logging.info(e)
#                 if con:
#                     con.close()
#                 logging.info('Connection closed!')
#         else:
#             logging.info(f'NOT PUSHING TO DB because TEST is {isTest}')
#         logging.info('DONE!')