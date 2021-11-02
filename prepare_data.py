"""
This file is used to generate custom image dataset on drowsiness.
"""

import cv2
import uuid
import os
import time


if __name__ == '__main__':
    # constant
    DATA_DIR = "./data/"
    LABELS = ['awake', 'drowsy', 'background']
    NUM_IMAGES = 20 # change number to suit your desire

    cap = cv2.VideoCapture(0)

    # loop through labels
    for label in LABELS:
        print(f"Collecting images for {label}")
        # time.sleep(5)
        
        # loop through images
        for img_num in range(NUM_IMAGES):
            print(f"Collecting images for {label}, image number {img_num}")
            
            ret, frame = cap.read()
            imgname =  os.path.join(DATA_DIR, label, label + '.' + str(uuid.uuid1()) + '.jpg')
            cv2.imwrite(imgname, frame)
            cv2.imshow('Image Collection', frame)
            
            # 2 seconds delay
            time.sleep(2)
        
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
    cap.release()
    cv2.destroyAllWindows()