import face_recognition
import cv2 as cv
import PIL.Image
import PIL.ImageDraw
import os

file_name = "CL.jpg"

unknown_image = face_recognition.load_image_file(file_name)
image = cv.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image) # detects all the faces in image
face_landmarks_list = face_recognition.face_landmarks(unknown_image)


# Drawing rectangles over the faces
for index, face_location in enumerate(face_locations):

    top, right, bottom, left = face_location
    print(top, right, bottom, left)


    k = face_landmarks_list[index]['right_eyebrow']
    bottom = face_landmarks_list[index]['right_eyebrow'][0][1]


    for k1 in k:
        if bottom > k1[1]:
            bottom = k1[1]

    k = face_landmarks_list[index]['left_eyebrow']
    lbottom = face_landmarks_list[index]['left_eyebrow'][0][1]

    for k1 in k:
        if lbottom > k1[1]:
            lbottom = k1[1]

    bottom = min(bottom,lbottom)

    cv.rectangle(image, (left, top), (right, bottom), (0,255,0), 3)
    cv.imshow("forehead_detected", image)
    cv.waitKey(0)
