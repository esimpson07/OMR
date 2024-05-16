import cv2 
import numpy as np

# Load image 
#image = cv2.imread('sheet-music.png')

#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image = cv2.imread(r"C:\Users\26simpsone\Downloads\sheetmusic.jpg")
'''#frame = cv2.medianBlur(frame, 3)
ret,thresh = cv2.threshold(frame,90,255,cv2.THRESH_BINARY)
thresh = cv2.medianBlur(thresh, 3)'''

# Set our filtering parameters 
# Initialize parameter setting using cv2.SimpleBlobDetector 
params = cv2.SimpleBlobDetector_Params() 

# Set Area filtering parameters 
params.filterByArea = True
params.minArea = 10
  
# Set Circularity filtering parameters 
params.filterByCircularity = True 
params.minCircularity = 0.5
  
# Set Convexity filtering parameters 
params.filterByConvexity = False
params.minConvexity = 0.2
      
# Set inertia filtering parameters 
params.filterByInertia = False
params.minInertiaRatio = 0.01
  
# Create a detector with the parameters 
detector = cv2.SimpleBlobDetector_create(params) 
      
# Detect blobs 
keypoints = detector.detect(image) 
  
# Draw blobs on our image as red circles 
blank = np.zeros((1, 1))  
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255), 
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
  
number_of_blobs = len(keypoints) 
text = "Number of Circular Blobs: " + str(len(keypoints)) 
cv2.putText(blobs, text, (20, 550), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2) 

# Show blobs 
cv2.imshow("Filtering Circular Blobs Only", blobs)

'''if cv2.waitKey(1) & 0xFF == ord('q'): 
    break'''

cv2.waitKey(0) 
cv2.destroyAllWindows()


