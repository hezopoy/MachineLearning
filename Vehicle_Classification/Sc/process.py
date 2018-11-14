import cv2
import numpy as np
def contains(r1,r2):
    x1,y1,w1,h1 = r1
    x2,y2,w2,h2 = r2
    if x1 < x2 and y1 < y2 and w1 > w2 and h1 > h2:
        return True
    return False
def getContours(im) :
    #im = cv2.imread("../Data/01/xedap1.jpg",0)
    #im2 = cv2.imread("../Data/01/xedap1.jpg",1)
    #im2 original image
    #(thresh,bw_im) = cv2.threshold(imgray,127,255,0)
    im = cv2.Canny(im,200,255)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(4,4))
                #im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
          #im = cv2.erode(im,kernel,iterations = 1)
    im= cv2.dilate(im,kernel,iterations = 1)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    _, contours, _= cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_rects = [cv2.boundingRect(ctr) for ctr in contours]
    rects_final = digit_rects[:]
    #loai bo nhung vat the kich thuoc nho va nhung vat the co chieu cao lon hon chieu rong
    for r in digit_rects:
        x,y,w,h = r
        if (w< 50 and h < 50) or (h>w):        
            rects_final.remove(r)
    """
    #loai bo nhung vat the nam ben trong
    for r1 in digit_rects:
           for r2 in digit_rects:
               if (r1[1] != 1 and r1[1] != 1) and (r2[1] != 1 and r2[1] != 1):           
                   if contains(r2,r1) and (r2 in rects_final):
                       rects_final.remove(r2)
    """
    print len(rects_final)
    #cv2.imshow("aa",crop_img)
    return rects_final

def getObjectData(rects,img):
    data = []
    for cnt in rects:
        x,y,w,h = cnt
        crop_img = img[y:y+h, x:x+w]
        crop_img = cv2.resize(np.array(crop_img),(70,50))
        data.append(crop_img)
    return data
def drawAllContours(im2,rects):
    """
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(im2, [box], 0 ,(0,255,0),2)
    cv2.imshow("aa",im2)
    #cv2.drawContours(imgray, contours, -1, (0,255,0), 3)
    """
    for cnt in rects:
        x,y,w,h = cnt
        cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,0),2)
    return im2
