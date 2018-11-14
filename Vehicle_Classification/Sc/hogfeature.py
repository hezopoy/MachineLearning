from skimage import exposure,data
from skimage import feature
import cv2
import numpy as np
def getFeature(img):
    #img 70x50 = 3500 feature
    #img = cv2.imread("Data/01/images0"+str(2)+".jpg",0)
    (H, hogImage)= feature.hog(img, orientations=9, pixels_per_cell=(5, 5),
            cells_per_block=(1, 1), visualize=True, block_norm="L1")
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
    #print len(H)
    return hogImage
"""
img = cv2.imread("Data/01/images0"+str(2)+".jpg",0)
im = cv2.resize(np.array(img),(70,50))
im = getFeature(im)
print np.size(im[0,:])

print np.size(im[:,0])
cv2.imshow("aaa",im)
print "done"
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
"""
