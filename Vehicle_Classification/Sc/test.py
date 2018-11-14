import numpy as np
import model
import cv2
import process
import hogfeature
io_path = "../Data/trainning/"
#maxval = 0
maxval  = np.genfromtxt(io_path + "maxval.csv",delimiter=",")
m1 = np.genfromtxt(io_path + "m1.csv",delimiter=",")
m2 = np.genfromtxt(io_path + "m2.csv",delimiter=",")
m3 = np.genfromtxt(io_path + "m3.csv",delimiter=",")
m1 = np.array([m1]).T
m2 = np.array([m2]).T
m3 = np.array([m3]).T
m= np.array([m1,m2,m3])
print "read dataset"
ne = 10
c = 0
for n in range(4):
      if n == 0:
            continue
      path = "../Data/DataTest/0" + str(n)
      count = 0
      for i in range(ne):
            if(i < 10):
                  img = cv2.imread(path+"/images0"+str(i)+".jpg",0)
            else:
                  img = cv2.imread(path+"/images"+str(i)+".jpg",0)
            rects = process.getContours(img)
            objects = process.getObjectData(rects,img)
            if(len(objects) == 1): 
                  feature = np.array(hogfeature.getFeature(objects[0]).reshape(1,3500).astype(np.float32))
            else:
                  continue
            [z,num] = model.euclidean_classifier(m,feature)
            if maxval < num:
                        z = -1
            if z == (n-1):
                  count += 1
                  """
                  if maxval < num:
                        maxval = num
                        print maxval
                        print i
                        c += 1
                  """
            else:
                  z = -1
            if(z == -1):
                  print "Sai"
                  print i
                  print "--------------------------"
                  c += 1
            
      print "cal"
      print count/float(ne)
#print c/float(26)
"""
print maxval
maxval = np.array([maxval])
np.savetxt(io_path + "maxval.csv",maxval+100,delimiter=",")
"""
print "Done"
