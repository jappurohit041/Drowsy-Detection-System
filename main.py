import PIL
from PIL import Image, ImageTk
import tkinter as tk
import argparse
import datetime
import cv2
import os
from scipy.spatial import distance
from imutils import face_utils
import dlib
import cv2
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np

class Application:
    def __init__(self, output_path = "./"):
        """ Initialize application which uses OpenCV + Tkinter. It displays
            a video stream in a Tkinter window and stores current snapshot on disk """
        self.vs = cv2.VideoCapture(0) # capture video frames, 0 is your default video camera
        self.output_path = output_path  # store output path
        self.current_image = None  # current image from the camera
        self.EAR_THRESH = 0.15
        self.MAR_THRESH= 0.20
        self.COUNTER_BLINK = 0
        self.COUNTER_YAWN = 0
        self.FRAME_CHECK = 20
        self.blink_flag = False
        self.flag = 0
        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor(".\shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code
        ret, self.frame = self.vs.read()
        #frame = imutils.resize(frame, width=450)
        self.size = self.frame.shape
        self.model_points = np.array([(0.0, 0.0, 0.0),
                                 (0.0, -330.0, -65.0),        
                                 (-225.0, 170.0, -135.0),     
                                 (225.0, 170.0, -135.0),      
                                 (-150.0, -150.0, -125.0),    
                                 (150.0, -150.0, -125.0)])
        
        focal_length = self.size[1]
        center = (self.size[1]/2, self.size[0]/2)
        
        self.camera_matrix = np.array([[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], dtype = "double")
        
        self.dist_coeffs = np.zeros((4,1))
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

        self.root = tk.Tk()  # initialize root window
        self.root.title("Drowsiness Detection System")  # set window title
        # self.destructor function gets fired when the window is closed
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.panel = tk.Label(self.root)  # initialize image panel
        self.panel.pack(padx=10, pady=10)
        self.root.config(cursor="arrow")
        #Disabling the maximize and minimize function
        self.root.resizable(0,0)

        # create a button, that when pressed, will take the current frame and save it to file
        btn = tk.Button(self.root, text="Snapshot!", command=self.take_snapshot)
        btn.pack(fill="both", expand=True, padx=10, pady=10)

        # start a self.video_loop that constantly pools the video sensor
        # for the most recently read frame
        self.video_loop()

    def eye_aspect_ratio(self,eye):
      A = distance.euclidean(eye[1], eye[5])
      B = distance.euclidean(eye[2], eye[4])
      C = distance.euclidean(eye[0], eye[3])
      ear = (A + B) / (2.0 * C)
      return ear

    def mouth_aspect_ratio(self,mouth):
        A = distance.euclidean(mouth[13], mouth[19])
        B = distance.euclidean(mouth[14], mouth[18])
        C = distance.euclidean(mouth[15], mouth[17])
        D = distance.euclidean(mouth[12], mouth[16])
        return (A+B+C)/(3*D)
    
    def pupilDetection(self,eye):
        Ax = (eye[0][0] + eye[3][0])/2
        Ay = (eye[0][1] + eye[3][1])/2
    
        Bx = (eye[1][0] + eye[4][0])/2
        By = (eye[1][1] + eye[4][1])/2
        
        Cx = (eye[2][0] + eye[5][0])/2
        Cy = (eye[2][1] + eye[5][1])/2
        
        X = int((Ax+Bx+Cx)/3)
        Y = int((Ay+By+Cy)/3)
        return (X,Y)

    def video_loop(self):
        """ Get frame from the video stream and show it in Tkinter """
        ok, self.frame = self.vs.read()  # read frame from video stream
        #frame = cv2.resize(frame, (1500,1000))
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        subjects = self.detect(gray, 0)
        for subject in subjects:
            shape = self.predict(gray, subject)
            shape = face_utils.shape_to_np(shape)#converting to NumPy Array
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            mouth = shape[self.mStart:self.mEnd]
            
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            mouthEAR = self.mouth_aspect_ratio(mouth)
            ear = (leftEAR + rightEAR) / 2.0
            image_points = np.array([
                                (shape[34][0], shape[34][1]),
                                (shape[8][0], shape[8][1]),
                                (shape[36][0], shape[36][1]),
                                (shape[45][0], shape[45][1]),
                                (shape[48][0], shape[48][1]),
                                (shape[54][0], shape[54][1])
                                ], dtype="double")
    
            (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, self.camera_matrix, self.dist_coeffs)
            for p in image_points:
                cv2.circle(self.frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(self.frame, [mouthHull], -1, (0, 255, 0), 1)
            cv2.drawContours(self.frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(self.frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.putText(self.frame, "EAR : {:.2f}".format(ear), (10, 75),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(self.frame, "MAR : {:.2f}".format(mouthEAR), (10, 95),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(self.frame, "BLINK : {:.0f}".format(self.COUNTER_BLINK), (10, 115),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if ear < self.EAR_THRESH or mouthEAR >= self.MAR_THRESH or  p2[1] > p1[1]*1.4 or p2[0] > p1[0]*1.4 :
              self.flag += 1
              if self.flag >= self.FRAME_CHECK or   p2[1] > p1[1]*1.4 or p2[0] > p1[0]*1.4:
                  cv2.putText(self.frame, "****************ALERT!****************", (100, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                  cv2.putText(self.frame, "****************ALERT!****************", (100,450),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                  pygame.mixer.init()
                  pygame.mixer.music.load("alarm.wav")
                  pygame.mixer.music.play()
              elif self.flag >= 2 and self.flag <= 10 and not self.blink_flag and not mouthEAR >= self.MAR_THRESH:
                  self.blink_flag = True
                  self.COUNTER_BLINK += 1
            else:
                cv2.circle(self.frame, self.pupilDetection(leftEye),2, (0, 0, 255), -1)
                cv2.circle(self.frame, self.pupilDetection(rightEye),2, ( 0, 0,255), -1)
                self.blink_flag = False
                self.flag = 0
            
        if ok:  # frame captured without any errors
            key = cv2.waitKey(1)
            cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            self.current_image = Image.fromarray(cv2image)  # convert image for PIL
            #self.current_image= self.current_image.resize([1280,1024],PIL.Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter 
            self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector  
            self.panel.config(image=imgtk)  # show the image
            #self.root.attributes("-fullscreen",True)
        self.root.after(20, self.video_loop)  # call the same function after 30 milliseconds

    def take_snapshot(self):
        """ Take snapshot and save it to the file """
        ts = datetime.datetime.now() # grab the current timestamp
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))  # construct filename
        p = os.path.join(self.output_path, filename)  # construct output path
        cv2.imwrite(p+'.jpg',self.frame)
        #self.current_image.save(p, "JPEG")  # save image as jpeg file
        print("[INFO] saved {}".format(filename))

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="./",
    help="path to output directory to store snapshots (default: current folder")
args = vars(ap.parse_args())

# start the app
print("[INFO] starting...")
pba = Application(args["output"])
pba.root.mainloop()
