import cv2

# Our Image
#img_file = 'dc2.jpg'
video = cv2.VideoCapture('carsxpeds.mp4')

# Our pretrained classifier
car_tracker_file = 'carsx.xml'
ped_tracker_file = 'haarcascade_fullbody.xml'

# create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
ped_tracker = cv2.CascadeClassifier(ped_tracker_file)

while True:
    # Read the current frame
    (read_successful, frame) = video.read()

    # safe coding
    if read_successful:
        #convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    # detect cars x pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    peds = ped_tracker.detectMultiScale(grayscaled_frame)

    # draw rectangles around cars
    for(x,y,w,h) in cars:
          cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 3)
    # draw rectangles around cars
    for(x,y,w,h) in peds:
          cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 3)

    #Display image
    cv2.imshow('Cars X Pedestrians', frame)

    # Do not autoclose
    key = cv2.waitKey(1)

    if key ==81 or key==113:
        break
video.release()
"""
# create opencv image
img = cv2.imread(img_file)

# convert to grayscale
bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars
cars = car_tracker.detectMultiScale(bw_img)

# draw rectangles around cars
for(x,y,w,h) in cars:
      cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 3)



#Display image
cv2.imshow('Cars X Pedestrians', img)

# Do not autoclose
cv2.waitKey()
"""

print('code ran successfully')