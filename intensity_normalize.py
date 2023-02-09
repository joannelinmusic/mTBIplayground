from locale import normalize
import numpy as np
import cv2
# import matplotlib.pyplot as plt
import os
import csv


# Normalizing images by type, running once will be only one type
def normalize_image(folder, type):
    images = []
    with open('min_max_intensity_record.csv', 'w', newline='') as f:
        fieldNames = ['patientID', 'image filename','Min Intensity(before)', 'Max Intensity(before)', 'Min Intensity(after)', 'Max Intensity(after)']
        myWriter = csv.DictWriter(f, fieldnames=fieldNames)
        myWriter.writeheader()

        for patientNumber in os.listdir(folder):
            if not patientNumber.startswith('.'):
                for image_type in os.listdir(folder+"/"+patientNumber):
                    if image_type == type:
                        print(image_type)
                        for last_folder in os.listdir(folder+"/"+patientNumber+"/"+image_type):
                            if not last_folder.startswith('.'):
                                for slices in os.listdir(folder+"/"+patientNumber+"/"+image_type+"/"+last_folder):

                                    img = cv2.imread(os.path.join(folder,patientNumber,image_type,last_folder,slices))
                                    
                                    if img is not None:
                                        
                                        print(patientNumber, slices, np.amin(img), np.amax(img))
                                        normalizedimage = cv2.normalize(img,None, 0, 255, cv2.NORM_MINMAX)
                                        if np.amin(img)!=np.amax(img) and np.mean(img) != 0:
                                            myWriter.writerow({'patientID': patientNumber, 'image filename': slices, \
                                                'Min Intensity(before)': np.amin(img), 'Max Intensity(before)': np.amax(img), \
                                                'Min Intensity(after)': np.amin(normalizedimage), 'Max Intensity(after)':np.amax(normalizedimage)})
                                            
                                        # Command lines to print out the min and max intensity
                                        # cv2.imshow('Normalized_image', normalizedimage)
                                        # cv2.waitKey(0)
                                        # cv2.destroyAllWindows()

                                        # print(np.amin(normalizedimage), np.amax(normalizedimage))
                                        
                                        # writeToCSV(path,patientNumber,image_type,xxx,yyy,zzz)
                                        # saveImage(path,"",nornalizedImage)
                                        # images.append(img)
                                    
    return images

# normalize_image("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/mTBI_data_new_2022/mTBI_Data_JPEGs", "T1")
# normalize_image("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/mTBI_data_new_2022/mTBI_Data_JPEGs", "T2")
normalize_image("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/mTBI_data_new_2022/mTBI_Data_JPEGs", "FLAIR")

