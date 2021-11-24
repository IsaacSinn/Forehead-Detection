import cv2 as cv
import dlib
import numpy as np
import threading
import math
import sys
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from scipy.integrate import quad

np.set_printoptions(threshold=sys.maxsize)

class forehead_detect():

    def __init__(self):

        dat_file = "shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dat_file)
        self.grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.grey_blur = cv.GaussianBlur(self.grey, (5,5), 0)
        self.kernel = np.ones((5,5),np.uint8)
        self.inter_pupil_distance = 60.3 # unit in mm
        self.forehead_length = 0
        self.mean = 60 # unit in mm
        self.sd = 8.89 # unit in mm


    @staticmethod
    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    @staticmethod
    def distance(ptA, ptB):
        return dist.euclidean(ptA, ptB)

    @staticmethod
    def nothing(self):
        pass

    @staticmethod
    def prop_density(x, mean, sd):
        prob_density = (1 / (math.sqrt(2*np.pi) * sd)) * np.exp(-0.5*((x-mean)/sd)**2)
        return prob_density

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
            y_step = int(midpoint_y - 10)
            while not hair_line and y_step >= 0:
                coordinate = np.array([x_subject(y_step), y_step])
                print(f"y_step: {y_step}, coordinate: {coordinate}, value: {self.edge[int(y_step)][int(x_subject(y_step))]}")


                if self.edge[int(y_step)][int(x_subject(y_step))] != [0]:
                    cv.circle(image, (int(x_subject(y_step)), int(y_step)), 2, (255,0,0), -1)
                    x_step = int(x_subject(y_step))
                    hair_line = True
                else:
                    y_step -= 1

            # detect pixel distance of forehead
            pixel_distance = self.distance((int(midpoint_x), int(midpoint_y)), (x_step, y_step))

            # detect pixel distance of inter pupil length
            midpoint_lefteye_x, midpoint_lefteye_y = map(int, self.midpoint(shape[37], shape[40]))
            midpoint_righteye_x, midpoint_righteye_y = map(int, self.midpoint(shape[43], shape[46]))
            inter_pupil_distance_pixel = self.distance((midpoint_lefteye_x, midpoint_lefteye_y), (midpoint_righteye_x, midpoint_righteye_y))

            # ratio to calculate length of forehead
            pixel_scale = self.inter_pupil_distance / inter_pupil_distance_pixel
            self.forehead_length = pixel_distance * pixel_scale
            print("forehead_length: ", self.forehead_length)


            # cv.imshow() display process
            cv.imshow("face_landmark", image)
            cv.imshow("edge", self.edge)
            # edge = plt.imshow(self.edge)
            # plt.show()

            # normal_distribution
            self.display_normal_distribution()



            if cv.waitKey(0) & 0xFF == 27:
                break

        cv.destroyAllWindows()

    def display_normal_distribution(self):
        prop_density = self.prop_density(self.forehead_length, self.mean, self.sd)

        area_probability = quad(self.prop_density, self.forehead_length, np.inf, args = (self.mean, self.sd))[0]
        area_probability *= 100

        # display the bell curve
        x = np.linspace(30,90,100)
        y= (1 / (math.sqrt(2*np.pi) * self.sd)) * np.exp(-0.5*((x-self.mean)/self.sd)**2)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.annotate(f"Top {round(area_probability, 1)}% percentile", (self.forehead_length, prop_density))
        ax.scatter(self.forehead_length, prop_density)

        plt.plot(x, y, "g", label = 'Normal Distribution Forehead Length')
        plt.text(70, 0.030, f"Forehead Length: {round(self.forehead_length, 1)}mm" , fontsize = 22, bbox = dict(facecolor = 'red', alpha = 0.5))
        plt.xlabel("Forehead Length", fontsize = 10)
        plt.ylabel("Prop Density",fontsize = 10)
        plt.legend(loc='upper left')
        plt.show()







    def adjust_parameter(self):
        cv.namedWindow("Canny Edge Parameters")
        cv.createTrackbar('High','Canny Edge Parameters',0,255,self.nothing)
        cv.createTrackbar('Low','Canny Edge Parameters',0,255, self.nothing)

        while(1):

            if cv.waitKey(1) & 0xFF == 27:
                break

            # get trackbar position
            High = cv.getTrackbarPos('High','Canny Edge Parameters')
            Low = cv.getTrackbarPos('Low','Canny Edge Parameters')

            # clean-up image
            self.edge = cv.Canny(self.grey_blur, Low, High)
            #self.edge = cv.morphologyEx(self.edge, cv.MORPH_OPEN, self.kernel)
            self.edge = cv.dilate(self.edge, self.kernel, iterations = 3)


            cv.imshow("edge", self.edge)

        cv.destroyAllWindows()



# driver code
if __name__ == '__main__':
    image =  cv.imread("chloe.png")

    forehead_detect = forehead_detect()
    forehead_detect.adjust_parameter()
    forehead_detect.detect(image)
