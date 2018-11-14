import cv2
def contains(r1,r2):
    x1,y1,w1,h1 = r1
    x2,y2,w2,h2 = r2
    if x1 < x2 and y1 < y2 and w1 > w2 and h1 > h2:
        return True
    return False
for i in range(13):
      if(i < 10):
            imgray = cv2.imread("../Data/DataTest/03/images0"+str(i)+".jpg",0)
      else:
            imgray= cv2.imread("../Data/DataTest/03/images"+str(i)+".jpg",0)
      im = cv2.Canny(imgray,200,255)
      kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(4,4))
            #im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
      im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
      #im = cv2.erode(im,kernel,iterations = 1)
      im= cv2.dilate(im,kernel,iterations = 1)
      im= cv2.dilate(im,kernel,iterations = 1)
      _, contours, _= cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      digit_rects = [cv2.boundingRect(ctr) for ctr in contours]
      rects_final = digit_rects[:]
      #loai bo nhung vat the kich thuoc nho va nhung vat the co chieu cao lon hon chieu rong
      #print (str(i) + ': ' + str(format(len(contours))))
      for r in digit_rects:
            x,y,w,h = r
            if (w< 50 and h < 50) or (h>w):        
                  rects_final.remove(r)
          #loai bo nhung vat the nam ben trong
      """
      for r1 in digit_rects:
            for r2 in digit_rects:
                  if (r1[1] != 1 and r1[1] != 1) and (r2[1] != 1 and r2[1] != 1):           
                        if contains(r2,r1) and (r2 in rects_final):
                              rects_final.remove(r2)      
      """
      print (str(i) + ': ' + str(format(len(rects_final))))
