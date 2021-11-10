import cv2 as cv
import os
import dlib
import numpy as np
import threading
from imutils import face_utils

image =  cv.imread("chloe_stretched.png")

class cv_read(threading.Thread):

    def __init__(self):

        self.cap = cv.VideoCapture(0)
        self.threading.Thread.__init__(self.get_video(), daemon = True)

    def get_video(self):
        while True:
            _, image = cap.read()

            cv.imshow("raw_image", image)

            if cv.waitKey(0) == 27:
                break

        cv.destroyAllWindows()
        cap.release()


class forehead_detect(threading.Thread):

    def __init__(self):

        dat_file = "shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dat_file)
        threading.Thread.__init__(self.detect(), daemon = True)

    def detect(self):
        while True:

            self.grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            self.rects = self.detector(self.grey, 0)

            for rect in self.rects:

                shape = self.predictor(self.grey, rect)

                # change into np array
                shape_np = np.zeros((68, 2), dtype="int")
                for i in range(0, 68):
                    shape_np[i] = (shape.part(i).x, shape.part(i).y)
                shape = shape_np

                for i, (x, y) in enumerate(shape):
                    cv.circle(image, (x,y), 1, (0,255,0), -1)


            cv.imshow("face_landmark", image)

            if cv.waitKey(0):
                break

        cv.destroyAllWindows()




if __name__ == '__main__':

    forehead_detect = forehead_detect()
    forehead_detect.start()
    #cv_read = cv_read()
    #cv_read.start()
