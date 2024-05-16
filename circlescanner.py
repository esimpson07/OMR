import cv2
import numpy as np

vid = cv2.VideoCapture(0) 
  
while(True): 
    ret, img = vid.read() 

    #img = cv2.imread("sheet-music.png")
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame = cv2.medianBlur(frame, 5)
    ret,thresh = cv2.threshold(frame,90,255,cv2.THRESH_BINARY)
    thresh = cv2.medianBlur(thresh, 3)

    params = cv2.SimpleBlobDetector_Params() 

    # Set Area filtering parameters 
    params.filterByArea = True
    params.minArea = 10
      
    # Set Circularity filtering parameters 
    params.filterByCircularity = True 
    params.minCircularity = 0.7
      
    # Set Convexity filtering parameters 
    params.filterByConvexity = False
    params.minConvexity = 0.2
          
    # Set inertia filtering parameters 
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
      
    # Create a detector with the parameters 
    detector = cv2.SimpleBlobDetector_create(params) 
          
    # Detect blobs 
    keypoints = detector.detect(thresh) 
      
    # Draw blobs on our image as red circles 
    blank = np.zeros((1, 1))  
    blobs = cv2.drawKeypoints(thresh, keypoints, blank, (0, 255, 0), 
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
      
    number_of_blobs = len(keypoints) 
    text = "Number of Circular Blobs: " + str(len(keypoints)) 
    cv2.putText(blobs, text, (20, 550), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2) 

    # Show blobs thresh = cv.medianBlur(thresh, 3)
    kernel = np.ones((5, 5), np.uint8)
    blobs = cv2.erode(blobs, kernel)
    cv2.imshow("Filtering Circular Blobs Only", blobs)

    '''try:
        circles = cv2.HoughCircles(thresh,cv2.HOUGH_GRADIENT,1,20,
            param1=30,param2=15,minRadius=0,maxRadius=50)

        circles = np.uint16(np.around(circles))
        
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

    except Exception as inst:
        print(type(inst))
        
    cv2.imshow('frame', img)'''

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release() 
cv2.destroyAllWindows()
