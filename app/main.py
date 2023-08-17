import base64
import cv2
import numpy as np
from fastapi import FastAPI
app = FastAPI()

@app.get('/')
def root():
    return{"hello"}

@app.get('/api/genhog')
def genhog(data_base64pixe: str):
    data_split_img = data_base64pixe.split(',',1)[1]
    
    decode_img_data = base64.b64decode(data_split_img)
    
    decode_img = cv2.imdecode(np.frombuffer(decode_img_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    
    resize = cv2.resize(decode_img,(128,128),cv2.INTER_AREA)

    win_size = resize.shape
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    # Set the parameters of the HOG descriptor using the variablesdefined above
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
    cell_size, num_bins)
    # Compute the HOG Descriptor for the gray scale image
    hog_descriptor = hog.compute(resize)
    hogvec = hog_descriptor.tolist()
    return ('HOG', hogvec)