# -*- coding: utf-8 -*-
import cv2

class LineTracking():
   
    def __init__(self, video_file):
        #"""The constructor."""
        self.video = cv2.VideoCapture(video_file)
        self.img_inter = None
        self.img_final = None
        self.centroids = []
        self.mean_centroids = [0, 0]

    def processing(self):
        #"""Méthode permettant le traitement d'image"""
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            
            # Processing on the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
            self.img_inter = thresh
            
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
            
            connectivity = 8
            output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
            num_labels = output[0]
            labels = output[1]
            stats = output[2]
            self.centroids = output[3]
            
            for c in self.centroids:
                self.mean_centroids[0] += c[0] / len(self.centroids)
                self.mean_centroids[1] += c[1] / len(self.centroids)

            self.img_final = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

            for c in self.centroids:
                self.img_final[int(c[1])-5: int(c[1])+10, int(c[0])-5: int(c[0])+10] = [0, 255, 0]

            cv2.imshow('process', self.img_inter)  # Display processed frame
            cv2.imshow('cable', self.img_final)    # Display final processed frame
            
            key = cv2.waitKey(1)
            if key == ord(' '): 
                break

        self.video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    test = LineTracking('2_ip.mp4')  # Replace '2_ip.mp4' with your video file path
    test.processing()
