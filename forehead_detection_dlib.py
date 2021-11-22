import cv2 as cv
import os
import dlib
import numpy as np
import threading
from imutils import face_utils
import math
import sys

np.set_printoptions(threshold=sys.maxsize)

class forehead_detect():

    def __init__(self):

        dat_file = "shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dat_file)
        self.grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.grey_blur = cv.GaussianBlur(self.grey, (5,5), 0)
        self.edge = cv.Canny(self.grey_blur, 50, 75)

    @staticmethod
    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    @staticmethod
    def distance(ptA, ptB):
        pass

    def detect(self, image):
        self.image_size = image.shape[:2]
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

            # midpoint between eye brows
            midpoint_x, midpoint_y = self.midpoint(shape[19], shape[24])
            cv.circle(image, (int(midpoint_x), int(midpoint_y)), 1, (0,255,0), -1)

            # find line function of perpendicular line eyebrow level
            line_gradient = (shape[24][1] - shape[19][1]) / (shape[24][0] - shape[19][0])
            perp_gradient = -1 / line_gradient
            y_intercept = midpoint_y - (perp_gradient * midpoint_x)
            x_subject = lambda y: (y - y_intercept) / perp_gradient

            # draw line
            increase_y = math.floor(self.image_size[1] - midpoint_y)
            cv.line(image, (int(midpoint_x), int(midpoint_y)), (int(x_subject(midpoint_y - increase_y)), int(midpoint_y - increase_y)), (0, 255, 0), 2)

            # detect hairline
            hair_line = False
            y_step = midpoint_y - 1
            while not hair_line and y_step >= 0:
                coordinate = np.array([x_subject(y_step), y_step])

                if self.edge[int(x_subject(y_step))][int(y_step)]:
                    cv.circle(image, (int(x_subject(y_step)), int(y_step)), 5, (0,255,0), -1)
                    hair_line = True
                else:
                    y_step -= 1
                    print(y_step)




            # cv.imshow() display process
            cv.imshow("face_landmark", image)
            cv.imshow("edge", self.edge)

            if cv.waitKey(0):
                break

        cv.destroyAllWindows()



# driver code
if __name__ == '__main__':
    image =  cv.imread("chloe_stretched.png")

    forehead_detect = forehead_detect()
    forehead_detect.detect(image)
    #cv_read = cv_read()
    #cv_read.start()
