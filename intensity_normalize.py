import numpy as np
import cv2
# import matplotlib.pyplot as plt
import os
# "/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/mTBI_data_new_2022/mTBI_Data_JPEGs"

def image_dataT1(folder):
    images = []
    for patientNumber in os.listdir(folder):
        if not patientNumber.startswith('.'):
            for image_type in os.listdir(folder+"/"+patientNumber):
                
                if image_type == "T1":
                    print(image_type)
                    print("yay1")
                    for last_folder in os.listdir(folder+"/"+patientNumber+"/"+image_type):
                        if os.path.isfile(os.path.join(folder+"/"+patientNumber+"/"+image_type, last_folder)):
                            img = cv2.imread(os.path.join(folder,patientNumber,image_type,last_folder))
                            if img is not None:
                                images.append(img)
                                print("yay")
    return images

image_dataT1("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/mTBI_data_new_2022/mTBI_Data_JPEGs")
# # cv2.imread(path, cv2.IMREAD_GRAYSCALE)
