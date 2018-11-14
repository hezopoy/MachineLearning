import numpy as np
import model
import cv2
import process
import hogfeature
import trainning
def scan(path):
      io_path = "../Data/trainning/"
      #path = "../Data/DataTest/04"
      m = trainning.readDataTrainning(io_path)
      maxval  = np.genfromtxt(io_path + "maxval.csv",delimiter=",")
      img = cv2.imread(path,0)
      imgout = cv2.imread(path,1)
      rects = process.getContours(img)
      objects = process.getObjectData(rects,img)
      for cnt in rects:
            x,y,w,h = cnt
            crop_img = img[y:y+h, x:x+w]
            crop_img = cv2.resize(np.array(crop_img),(70,50))
            feature = np.array(hogfeature.getFeature(crop_img).reshape(1,3500).astype(np.float32))
            [z,num] = model.euclidean_classifier(m,feature)
            print z
            print num
            if maxval < num:
                  continue
            if z==0:
                  cv2.rectangle(imgout,(x,y),(x+w,y+h),(255,0,0),2)
            if z==1:
                  cv2.rectangle(imgout,(x,y),(x+w,y+h),(0,0,255),2)
            if z==2:
                  cv2.rectangle(imgout,(x,y),(x+w,y+h),(0,255,0),2)
      imgout = cv2.resize(np.array(imgout),(500,500))
      cv2.imwrite("detect.jpg",imgout)
      return
