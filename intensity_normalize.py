from locale import normalize
import numpy as np
import cv2
# import matplotlib.pyplot as plt
import os

def image_dataT1(folder, type):
    images = []
    for patientNumber in os.listdir(folder):
        if not patientNumber.startswith('.'):
            for image_type in os.listdir(folder+"/"+patientNumber):
                if image_type == type:
                    print(image_type)
                    for last_folder in os.listdir(folder+"/"+patientNumber+"/"+image_type):
                        if not last_folder.startswith('.'):
                            for slices in os.listdir(folder+"/"+patientNumber+"/"+image_type+"/"+last_folder):

                                img = cv2.imread(os.path.join(folder,patientNumber,image_type,last_folder,slices))
                                print(img)
                                if img is not None:
                                    normalizedimage = cv2.normalize(img,None, 0, 255, cv2.NORM_MINMAX)
                                    cv2.imshow('Normalized_image', normalizedimage)
                                    cv2.waitKey(0)
                                    cv2.destroyAllWindows()
                                    
                                    # writeToCSV(path,patientNumber,image_type,xxx,yyy,zzz)
                                    # saveImage(path,"",nornalizedImage)
                                    # images.append(img)
                                    print("hehe")
    return images

image_dataT1("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/mTBI_data_new_2022/mTBI_Data_JPEGs", "T1")
# # cv2.imread(path, cv2.IMREAD_GRAYSCALE)
