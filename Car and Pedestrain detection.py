import cv2

video = cv2.VideoCapture(r'C:\Users\tejas\PycharmProjects\pythonProject1\carr.mp4')

# Importing the Models
car_cascade = cv2.CascadeClassifier(r'C:\Users\tejas\PycharmProjects\pythonProject1\cars1.xml')
pede_cascade = cv2.CascadeClassifier(r'C:\Users\tejas\PycharmProjects\pythonProject1\pedestrian.xml')

flag = True

while flag:
    ret,frames = video.read()

    #Converting the frames of the video to grayscale
    gray_image = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    #Scaling the image
    cars = car_cascade.detectMultiScale(gray_image, 1.1, 2)
    pedestrian = pede_cascade.detectMultiScale(gray_image, 1.1, 1)

    # Drawing the Boundary Box
    for x, y, w, h in cars:
        cv2.rectangle(frames, (x,y),(x+w,y+h),(255,0,0),2)
        for x, y, w, h in pedestrian:
            cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Visualising the Results
    cv2.imshow('Video', frames)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()