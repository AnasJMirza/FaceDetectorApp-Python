import cv2 as cv
from random import randrange

# #  Reading the image to detect face
# img = cv.imread('./images/test.jpg')

# #  Must convert it into grayScale
# grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# # load some pre-trained data on frontal faces
# trainedFaceData = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# # detect faces
# faceCoordinates = trainedFaceData.detectMultiScale(grayimg)
# print(faceCoordinates)
# for (x, y, w, h) in faceCoordinates:
#     cv.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)


# #  for showing
# cv.imshow("Window", img)
# cv.waitKey()

# print("hello")

webcam = cv.VideoCapture(0)

while True :
    # Read the current frame
    successfulFrameRead, frame = webcam.read()

    # changr to gray scale
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # load some pre-trained data on frontal faces
    trainedFaceData = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    # detect faces
    faceCoordinates = trainedFaceData.detectMultiScale(grayFrame)
    

    # drawing rectangles around the faces
    for (x, y, w, h) in faceCoordinates:
        cv.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

    # showing realTime video
    cv.imshow("video", frame)
    key = cv.waitKey(1)

    # Stop when the key is pressed
    if key == 81 or key == 113:
        break

# release the webcam object
webcam.release()
