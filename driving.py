import cv2

# our video
video = cv2.VideoCapture(0)

# our trained data
car_trained_data = 'cars.xml'
pedestrains_trained_data = 'pedestrains.xml'

# our classifier
car_data = cv2.CascadeClassifier(car_trained_data)
pedestrains_data = cv2.CascadeClassifier(pedestrains_trained_data)


while True:
   # read each frame
   read_successful, frame = video.read()

   # safe code
   if read_successful:
      # change video color to gray
      grayscaled_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   else:
      break

   # detects the frames in gray for cars and pedestrains
   car = car_data.detectMultiScale(grayscaled_vid)
   pedestrains = pedestrains_data.detectMultiScale(grayscaled_vid)

   # this is for car coordinates detection
   for (x, y, w, h) in car:
      cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 0), 2)

   # pedestrains coordinates
   for (x, y, w, h) in pedestrains:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

   # displays the frame
   cv2.imshow('Cars and Pedestrains', frame)

   # waits before displaying the next frame
   Key = cv2.waitKey(1)

   if Key==81 or Key==113:
      break

print("Code Complete")
