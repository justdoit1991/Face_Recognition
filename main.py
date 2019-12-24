# import module
import os
import cv2
import math
import numpy as np
import pandas as pd
from os import listdir
from mtcnn.mtcnn import MTCNN
from src.img_resize import img_resize
from src.feature_extractor import FeatureExtractModel
from sklearn.metrics.pairwise import cosine_similarity

# load model
detector = MTCNN()
model_path = os.getcwd() + '\\model\\facenet.pb'
fe = FeatureExtractModel(model_path, gpu_memory_fractio=0.6)

'''
# read picture_base
# rename columns
#picture_base = pd.read_csv(os.getcwd() + '\\picture_base.csv', index_col=0)
#picture_base.columns = pd.RangeIndex(start=0, stop=512, step=1)

## 1.create picture base & 2.added new picture
# decide action
action = 2 # (1 = create picture base , 2 = added new picture)

# list all files
if action == 1:
    files_path = os.getcwd() + '\\data\\'
elif action == 2:
    files_path = os.getcwd() + '\\added\\'

files = listdir(files_path)

# load image and resize to model input size 
data = pd.DataFrame()
for img_name in files:
          
    img = cv2.imread(files_path + img_name)  # load image(avoid chinese path)
    
    # avoid image too big
    if img.size > 480*640*3:
        img2 = cv2.resize(img, (480, round(480*(img.shape[0]/img.shape[1]))), interpolation = cv2.INTER_AREA)
        reduce_ratio = img.shape[0] / img2.shape[0]  # reduce ratio 
    else:
        img2 = img.copy()
        reduce_ratio = 1
    
    face_list = detector.detect_faces(img2)   # face detect and alignment
   
    x, y, w, h = [round(x * reduce_ratio) for x in face_list[0]['box']]  # find the boundingbox        
    infer_img = img_resize(img, x, y, w, h)    # extract face and resize      
    emb = pd.DataFrame(fe.infer([infer_img]))  # use [] add dimension and infer
 
    emb.index = [img_name.split('.')[0]]  # Let img_name as index
    data = data.append(emb)               # comb all embeddings vector
    
if action == 1:
    picture_base = data
elif action == 2:
    picture_base = picture_base.append(data)  
    
# save picture_base
#picture_base.to_csv(os.getcwd() + '\\picture_base.csv') 
'''

## 3.who is he/she?
test_img = 'test.jpg'
img = cv2.imread(os.getcwd() + '\\test\\' + test_img)  # load image(avoid chinese path)

# avoid image too big
if img.size > 480*640*3:
    img2 = cv2.resize(img, (480, round(480*(img.shape[0]/img.shape[1]))), interpolation = cv2.INTER_AREA)
    reduce_ratio = img.shape[0] / img2.shape[0]  # reduce ratio 
else:
    img2 = img.copy()
    reduce_ratio = 1
    
face_list = detector.detect_faces(img2)  # face detect and alignment
    
# recognize one or more people
for i in range(0, len(face_list)):
       
    x, y, w, h = [round(x * reduce_ratio) for x in face_list[i]['box']]  # find the boundingbox  
    infer_img = img_resize(img, x, y, w, h)    # extract face and resize       
    emb = pd.DataFrame(fe.infer([infer_img]))  # use [] add dimension and infer

    # use cosine distance cross comparison  
    cos_dist = pd.DataFrame(np.arccos(cosine_similarity(picture_base, emb)) / math.pi)
    cos_dist.index = picture_base.index        
    cos_dist = cos_dist.reset_index().rename(columns = {"index": "name", 0: "dist"}).sort_values(['dist'], ascending=[1]).reset_index(drop = True)

    # what's his/her name (threshold = 0.3)
    if cos_dist.loc[0, 'dist'] <= 0.3:
        name = cos_dist.loc[0, 'name']
    else:
        name = 'unknown'

    # auto adjust width and font Scale
    # draw boundingbox (color = (blue, green, red) , width)
    # putText(image, text, org, font, fontScale, color, Line thickness, Line kinds)
    width = fontScale = thickness = math.ceil(img.size / 4000000)
    cv2.rectangle(img, (x,y-5), (x+w,y+h), (255,0,0), width)
    cv2.putText(img, name, (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL,  
                fontScale, (0, 255, 0), thickness, cv2.LINE_AA)   

# save image
cv2.imwrite("C:/Users/Felix/Desktop/result.jpg", img)
