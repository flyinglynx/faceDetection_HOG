from tensorflow.keras import models
import numpy as np
import cv2
import lib.HOG as hog

class FaceDetector():
    def __init__(self,model_path,w=200,h=200):
        self.detect_model = models.load_model(model_path)
        self.w = w     #图片缩放后的宽度
        self.h = h     #图片缩放后的高度
        
    def detect(self,img):
        img = cv2.resize(img, (self.w,self.h),interpolation = cv2.INTER_AREA)
        hog_des = hog.Hog_descriptor(img,cell_width=16, block_width=3, bin_size=8, block_stride=1, gamma=3.0)
        hog_vec = hog_des.extract()
        hog_vec = np.array([hog_vec])
        predict = self.detect_model.predict(hog_vec)
        return predict
    