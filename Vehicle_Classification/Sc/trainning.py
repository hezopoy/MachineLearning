import numpy as np
import cv2
import process
import hogfeature
import model
def setData(path,number):
      j = 0
      featureData = []
      for i in range(number):
            if(i < 10):
                  img = cv2.imread(path+"/images0"+str(i)+".jpg",0)
            else:
                  img = cv2.imread(path+"/images"+str(i)+".jpg",0)
            rects = process.getContours(img)
            objects = process.getObjectData(rects,img)
            if(len(objects) == 1):
                  #img 70x50 = 3500 pixel
                  feature = np.array(hogfeature.getFeature(objects[0]).reshape(3500,1).astype(np.float16))
                  featureData.append(feature)
                  print "setImg "+str(j)+":Done"
                  j = j + 1
      return featureData      

def writeDataTrainning(io_path,n):
      path = "../Data/01"
      print "setDataClass1"
      featureDataClass1 = setData(path,n)
      featureDataClass1 = np.array(featureDataClass1)

      path = "../Data/02"
      print "setDataClass2"
      featureDataClass2 = setData(path,n)
      featureDataClass2 = np.array(featureDataClass2)

      path = "../Data/03"
      print "setDataClass3"
      featureDataClass3 = setData(path,n)
      featureDataClass3 = np.array(featureDataClass3)

      print "DataClass1"
      #print np.size(featureDataClass1[0,:])
      #print np.size(featureDataClass1[:,0])
      [m1,S1] = model.Gaussian_ML_estimate(featureDataClass1)
      np.savetxt(io_path + "m1.csv",m1,delimiter=",")
      np.savetxt(io_path + "S1.csv",S1,delimiter=",")
      print "Done"

      print "DataClass2"
      #print np.size(featureDataClass2[0,:])
      #print np.size(featureDataClass2[:,0])
      [m2,S2] = model.Gaussian_ML_estimate(featureDataClass2)
      np.savetxt(io_path + "m2.csv",m2,delimiter=",")
      np.savetxt(io_path + "S2.csv",S2,delimiter=",")
      print "Done"

      print "DataClass3"
      #print np.size(featureDataClass3[0,:])
      #print np.size(featureDataClass3[:,0])
      [m3,S3] = model.Gaussian_ML_estimate(featureDataClass3)
      np.savetxt(io_path + "m3.csv",m3,delimiter=",")
      np.savetxt(io_path + "S3.csv",S3,delimiter=",")
      print "Done"

      m = np.array([m1,m2,m3])
      S = np.array([S1,S2,S3])
      """
      print "m:"
      print m
      print "S:"
      print S
      """
      return [m,S]

def readDataTrainning(io_path):
      m1 = np.genfromtxt(io_path + "m1.csv",delimiter=",")
      #S1 = np.genfromtxt(io_path + "S1.csv",delimiter=",")
      m2 = np.genfromtxt(io_path + "m2.csv",delimiter=",")
      #S2= np.genfromtxt(io_path + "S2.csv",delimiter=",")
      m3 = np.genfromtxt(io_path + "m3.csv",delimiter=",")
      #S3 = np.genfromtxt(io_path + "S3.csv",delimiter=",")
      m1 = np.array([m1]).T
      m2 = np.array([m2]).T
      m3 = np.array([m3]).T
      m= np.array([m1,m2,m3])
      #S = np.array([S1,S2,S3])
      return m

io_path = "../Data/trainning/"
"""
print "read trainning data"
[m,S] = readDataTrainning(io_path)
print "done"

print "read dataset1"
img1 = cv2.imread("../Data/01/images41.jpg")
rects1 = process.getContours(img1)
objects1 = process.getObjectData(rects1,img1)
if(len(objects1) == 1):
      feature1 = np.array(hogfeature.getFeature(objects1[0]).reshape(1,3500).astype(np.float32))
print "done"
print "read dataset2"
img2 = cv2.imread("../Data/02/images41.jpg")
rects2 = process.getContours(img2)
objects2 = process.getObjectData(rects2,img2)
if(len(objects2) == 1):
      feature2 = np.array(hogfeature.getFeature(objects2[0]).reshape(1,3500).astype(np.float32))
print "done"
print "read dataset3"
img3 = cv2.imread("../Data/03/images27.jpg")
rects3 = process.getContours(img3)
objects3= process.getObjectData(rects3,img3)
if(len(objects3) == 1):
      feature3 = np.array(hogfeature.getFeature(objects3[0]).reshape(1,3500).astype(np.float32))
print "done"
print "euclidean xe dap"
print model.euclidean_classifier(m,feature1)
print "euclidean xe tay ga"
print model.euclidean_classifier(m,feature2)
print "euclidean xe so"
print model.euclidean_classifier(m,feature3)
"""
#[m,S] = writeDataTrainning(io_path,50)
print "Done"

