'''
ECE276A WI20 HW1
Stop Sign Detector
'''

import os, cv2
from skimage.measure import label, regionprops
import numpy as np
from matplotlib import pyplot as plt

class StopSignDetector():

        
        def __init__(self):
                '''
                        Set classifier parameter
                '''
                self.w = np.array([ -37532.8884858,  -171158.85021479 ,-185771.03081237 , 192951.91197725]).T

        def segment_image(self, img):
                '''
                        Obtain a segmented image using a logistic regression color classifier,
                        
                        Inputs:
                                img - original image
                        Outputs:
                                mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
                '''
        
                img_width = img.shape[0]
                img = img / 255
                # reshape image to get column of pixels and add column of ones
                flattened_image = img.reshape((-1,3))
                flattened_image = np.hstack([np.ones([flattened_image.shape[0], 1]), flattened_image])
                # segment image based on decistion boundary x^T w
                mask_img = ((flattened_image @ self.w) >= 0)
                # reshape to original image shape
                mask_img = mask_img.reshape((img_width, -1)).astype(np.float32) 
                
                return mask_img

        def get_bounding_box(self, img):
                '''
                        Find the bounding box of the stop sign
                        call other functions in this class if needed
                        
                        Inputs:
                                img - original image
                        Outputs:
                                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
                                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
                                is from left to right in the image.
                                
                        Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
                '''
                # Segment image
                mask_img = self.segment_image(img)

                # Remove small contours from image
                contours, _ = cv2.findContours(mask_img.astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                        if cv2.contourArea(cnt) < ((img.shape[0] * img.shape[1]) / 1000):
                                cv2.fillPoly(mask_img, pts =[cnt], color=(0,0,0))

                # get edges and perform dilation operation
                edges = cv2.Canny(mask_img.astype(np.uint8), 0, 0.5)
                
                elips = (3,3) 
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,elips)
                dilated = cv2.dilate(edges, kernel)
                contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                boxes = []

                for cnt in contours:
                        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
                        
                        if len(approx) > 6:
                                   
                                x,y,w,h = cv2.boundingRect(cnt)


                                if (np.abs(w - h) < 0.50 * w) and w * h > 20:

                                        x1 = x
                                        y1 = img.shape[0] - (y + h)
                                        x2 = x + w
                                        y2 = img.shape[0] - y

                                        boxes.append([x1, y1, x2, y2])
    

                # sort box list by x1 value in increasing order                       
                boxes = sorted(boxes, key=lambda x: x[0])
                
                return boxes

        def draw_boxes(self, img, boxes):
                '''
                        Draw bounding boxes on image

                        Inputs:
                                img - image to draw on
                                boxes - list of lists of box coordinates in xy-coordinates

                        Outputs:
                                img - image with boxes drawn on top

                '''

                
                # Green in RGB 
                color = (0, 255, 0) 
                  
                # Line thickness of 2 px 
                thickness = 2

                # Draw boxes
                for box in boxes:
                        if len(box) != 0:
                        
                                start_point = (box[0], img.shape[0] - box[1])

                                end_point = (box[2], img.shape[0] - box[3])

                                img = cv2.rectangle(img, start_point, end_point, color, thickness)

                return img


if __name__ == '__main__':
        folder = "trainset"
        my_detector = StopSignDetector()
        for filename in os.listdir(folder):
                if filename != '.DS_Store':
                        print('File: ', filename)
                        # read one test image
                        img = cv2.imread(os.path.join(folder,filename))
                        
                        cv2.imshow('image', img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                        #Display results:
                        #(1) Segmented images
                        mask_img = my_detector.segment_image(img)
                        #(2) Stop sign bounding box
                        boxes = my_detector.get_bounding_box(img)
                        #The autograder checks your answers to the functions segment_image() and get_bounding_box()
                        #Make sure your code runs as expected on the testset before submitting to Gradescope
                        
                        img_box = my_detector.draw_boxes(img, boxes)
                





