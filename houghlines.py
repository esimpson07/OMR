import sys
import math
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0) #opens a video stream

while True:
    ret, src = cap.read() #reads the frame from the camera

    '''The code below does the following:
     - Converts to grayscale, blurs the image to remove noise, and
     uses a threshold to filter the image to detect writing on a
     page. Then, a secondary blur is applied, and based off the average
     values of the image, a lower and upper value for the edge filter
     is used. Then, the image is dilated and eroded, and the edges are
     finally filtered based off this filtered image, to find the staff.'''
    
    imgray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    im_gauss = cv.GaussianBlur(imgray, (5, 5), 0)
    ret, thresh = cv.threshold(im_gauss, 127, 255, 0)
    
    blur = cv.blur(src, (3, 3))
    sigma = np.std(blur)
    mean = np.mean(blur)
    lower = int(max(0, (mean - sigma)))
    upper = int(min(255, (mean + sigma)))
    dst = cv.Canny(blur, lower, upper)
    kernel = np.ones((7,7), np.uint8)
    pdst = cv.dilate(dst, kernel)
    pdst = cv.dilate(pdst, kernel)
    pdst = cv.dilate(pdst, kernel)
    pdst = cv.erode(pdst, kernel)
    dst = cv.Canny(pdst, 0, 255)
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    '''The code below completes the following tasks:
     - Uses the HoughLinesP method to accurately detect all the straight
     lines in the image, with a length of 100 pixels or longer, requiring
     that the maximum gap between a line is 10 pixels.
     - Next, the lines get sorted in the lines_sort = sorted(...) lambda
     method. This sorts the lines from the lowest y value, to the greatest,
     such as y = [0, 100, 200, 300]. This allows the comparison between 2
     lines parallel to each other in order to detect notes.'''

    lines = cv.HoughLinesP(dst,1,np.pi/180,70,minLineLength=100,maxLineGap=20)
    if lines is not None:
        lines_sort = sorted(lines, key=lambda a_entry: a_entry[..., 1])
        for line in lines_sort:
            x1,y1,x2,y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            angle = math.atan(slope)
            xavg = (x1 + x2) / 2
            yavg = (y1 + y2) / 2
            xlen = x2 - x1
            ylen = y2 - y1
            print(yavg)
            
            cv.line(cdstP,(x1,y1),(x2,y2),(0,255,0),2)


    frame = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    frame = cv.medianBlur(frame, 5)
    ret,thresh = cv.threshold(frame,100,255,cv.THRESH_BINARY)
    

    params = cv.SimpleBlobDetector_Params() 

    #Set Area filtering parameters 
    params.filterByArea = True
    params.minArea = 10
      
    #Set Circularity filtering parameters 
    params.filterByCircularity = True 
    params.minCircularity = 0.7
      
    #Set Convexity filtering parameters 
    params.filterByConvexity = False
    params.minConvexity = 0.2
          
    #Set inertia filtering parameters 
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
      
    #Create a detector with the parameters 
    detector = cv.SimpleBlobDetector_create(params) 
          
    #Detect blobs 
    keypoints = detector.detect(thresh) 
      
    #Draw blobs on our image as red circles 
    blank = np.zeros((1, 1))  
    blobs = cv.drawKeypoints(thresh, keypoints, blank, (0, 0, 255), 
                              cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
      
    number_of_blobs = len(keypoints) 
    text = "Number of Circular Blobs: " + str(len(keypoints)) 
    cv.putText(blobs, text, (20, 550), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    #Puts the blob number on the filtered page

    
    #cv.imshow("Source", src)
    #cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("canny", dst)
    cv.imshow("pdst", pdst)
    cv.imshow("Detected Lines - Probabilistic Line Transform", cdstP)
    cv.imshow("Blobs", blobs)
    #cv.imshow("thresh", thresh)
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break
    
    
cap.release() 
cv.destroyAllWindows()
    
