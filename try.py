import pandas as pd
import csv
import matplotlib.pyplot as plt

# Define dataset to Pandas
csv_path = "/Users/joannelin/Desktop/Motorola/mTBIplayground/predictions(30_epoch_result).csv"
df = pd.read_csv(csv_path)

# num_of_img_per_patient = df.loc[df['tag'] == (0)].groupby('patientID').size().reset_index(name='Number of Images')

# num_of_img_per_patient_all = num_of_img_per_patient.loc[df[column_name].notnull()]
# print("Max:", num_of_img_per_patient.max(), "Min:", num_of_img_per_patient.min())

# Count number of images per patient in T1 (yes)
num_of_img_per_patient_yes = df.loc[df['tag'] == (1)].groupby('patientID').size().reset_index(name='Number of Images')
print("Max:", num_of_img_per_patient_yes['Number of Images'].max(), "Min:", num_of_img_per_patient_yes['Number of Images'].min())
# print(num_of_img_per_patient)

###########################

# csv_path_accuracyEpoch = "/Users/joannelin/Desktop/Motorola/mTBIplayground/accuracy_epoch.csv"

# train_accuracies = []
# val_accuracies = []
# test_accuracies = []
# epochs = []

# with open(csv_path_accuracyEpoch, newline='') as csvfile:
#     reader = csv.DictReader(csvfile)
#     train_accuracies.append(0)
#     val_accuracies.append(0)
#     test_accuracies.append(0)
#     epochs.append(0)
#     for row in reader:
#         train_accuracies.append(row["Train_accuracy"])
#         val_accuracies.append(row["Validation_accuracy"])
#         test_accuracies.append(row["Test_accuracy"])
#         epochs.append(row["Epoch"])
        
# df = pd.DataFrame({
#     "Epochs": epochs,
#     "Train_accuracies": train_accuracies,
#     "Val_accuracies": val_accuracies,
#     "Test_accuracies": test_accuracies
# })

# df["Train_accuracies"] = df["Train_accuracies"].astype(float)
# df["Val_accuracies"] = df["Val_accuracies"].astype(float)
# df["Test_accuracies"] = df["Test_accuracies"].astype(float)
# df["Epochs"] = df["Epochs"].astype(float)
# print(df["Epochs"])

# # print(df)

# ax = df.plot(x="Epochs", y=["Train_accuracies", "Val_accuracies", "Test_accuracies"], figsize=(8, 6))
# ax.set_xlabel("Epochs")
# ax.set_ylabel("Accuracy")
# ax.set_xlim([-1, 31])
# ax.set_ylim([0, 1])
# ax.set_title("Training, validation, testing accuracy over epoch")
# ax.legend(loc="best")
# plt.show()



# print(train_accuracies)
# print(val_accuracies)
# print(test_accuracies)
# plt.plot(train_accuracies, c='red',label="Train accuracy" )
# plt.plot(val_accuracies, c='green', label="Val accuracy")
# plt.plot(test_accuracies, c='blue', label="Test accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend(loc="upper left")
# plt.ylim(bottom=0.0, top=1.0)
# plt.show()

# print(tr_loss)
# print(val_loss)
# print(test_loss)
# plt.figure()
# plt.plot(tr_loss, c='red',label="Train loss" )
# plt.plot(val_loss, c='green', label="Val loss")
# plt.plot(test_loss, c='blue', label="test loss")
# # plt.plot(test_loss, c='blue', label="Test loss")
# plt.xlabel("Epochs")
# plt.ylabel("loss")
# plt.legend(loc="lower left")
