import csv
import os
from shutil import copyfile
import random
import splitfolders

# specify the path to the CSV file containing image names and categories
csv_path = "/Users/joannelin/Desktop/Motorola/mTBIplayground/classification_tag_T1.csv"

# specify the path to the directory containing the images
image_dir = "/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/mTBI_data_new_2022/mTBI_Data_JPEGs"
root_dir = "/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/yes_no_split"

# specify the names of the two directories you want to create
# train_dir = os.path.join(root_dir, "train")
# val_dir = os.path.join(root_dir, "validate")
# test_dir = os.path.join(root_dir, "test")

# # create the two directories if they don't already exist
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(val_dir, exist_ok=True)


# for directory in ["yes", "no"]:
#     train_subdir = os.path.join(train_dir, directory)
#     val_subdir = os.path.join(val_dir, directory)
#     test_subdir = os.path.join(test_dir, directory)
#     os.makedirs(train_subdir, exist_ok=True)
#     os.makedirs(val_subdir, exist_ok=True)
#     os.makedirs(test_subdir, exist_ok=True)


# read the CSV file and loop through its rows
# with open(csv_path, 'r') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     next(csv_reader)  # skip the header row
#     for row in csv_reader:
#         patient, image_name, binary_category = row[0], row[1], row[6]
#         for patientNumber in os.listdir(image_dir):
#             if patientNumber == patient:
#                 for image_type in os.listdir(image_dir+"/"+patientNumber):
#                     if image_type == "T1":
#                         for last_folder in os.listdir(image_dir+"/"+patientNumber+"/"+image_type):
#                             if not last_folder.startswith('.'):
#                                 for slices in os.listdir(image_dir+"/"+patientNumber+"/"+image_type+"/"+last_folder):
#                                     if slices == image_name:
                                        
                                        
#                                         if random.random() < 0.7:  # 70% of images go to the train set
#                                             dest_dir = os.path.join(train_dir)
#                                         elif random.random() < 0.8:  # 10% of images go to the validation set
#                                             dest_dir = os.path.join(val_dir)
#                                         else:  # 20% of images go to the test set
#                                             dest_dir = os.path.join(test_dir)
                                        
#                                         if row[6] == "Yes":
#                                             dest_dir = os.path.join(dest_dir, "yes")
#                                         elif row[6] == "No":
#                                             dest_dir = os.path.join(dest_dir, "no")
#                                         else:
#                                             continue
                                      
#                                         src_path = os.path.join(image_dir, patientNumber, image_type, last_folder, image_name)
#                                         dest_path = os.path.join(root_dir, dest_dir, image_name)
#                                         # copy the image from the source to the destination directory
#                                         copyfile(src_path, dest_path)



yes_dir = os.path.join(root_dir, "yes")
no_dir = os.path.join(root_dir, "no")
os.makedirs(yes_dir, exist_ok=True)
os.makedirs(no_dir, exist_ok=True)



# aread the CSV file and loop through its rows
with open(csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # skip the header row
    for row in csv_reader:
        patient, image_name, binary_category = row[0], row[1], row[6]
        for patientNumber in os.listdir(image_dir):
            if patientNumber == patient:
                for image_type in os.listdir(image_dir+"/"+patientNumber):
                    if image_type == "T1":
                        for last_folder in os.listdir(image_dir+"/"+patientNumber+"/"+image_type):
                            if not last_folder.startswith('.'):
                                for slices in os.listdir(image_dir+"/"+patientNumber+"/"+image_type+"/"+last_folder):
                                    
                                    if slices == image_name:
                                        
                                        
                                        if row[6] == "Yes":
                                            dest_dir = os.path.join(yes_dir)
                                        elif row[6] == "No":
                                            dest_dir = os.path.join(no_dir)
                                        else:
                                            continue
                                        

                                        
                                            
                                        src_path = os.path.join(image_dir, patientNumber, image_type, last_folder, image_name)
                                        dest_path = os.path.join(root_dir, dest_dir, image_name)
                                        # copy the image from the source to the destination directory
                                        if os.path.exists(dest_path):
                                            dest_path = os.path.join(root_dir, dest_dir, "_1"+image_name)
                                        copyfile(src_path, dest_path)

    splitfolders.ratio(root_dir, output=root_dir, seed=1337, ratio=(0.7, 0.1,0.2)) 
