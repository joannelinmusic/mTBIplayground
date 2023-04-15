from cv2 import transform
from sklearn.metrics import confusion_matrix
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import pandas as pd
from sklearn.metrics import confusion_matrix
import csv
import os
from PIL import Image

# root_dir = "/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/yes_no_split"

# for image in os.listdir(root_dir+"/train/yes"):
#     img = cv2.imread(os.path.join(root_dir,"train", "yes", image))
#     print(img.shape)


transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

# Define the dataset
train_set = datasets.ImageFolder("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/yes_no_split/train", transform = transformations)
val_set = datasets.ImageFolder("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/yes_no_split/val", transform = transformations)


# Define the batch size
batch_size = 30
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size =batch_size, shuffle=True)
    

class MyNetwork(nn.Module):
    #def __init__(self, num_classes=2):
    def __init__(self): 
        super(MyNetwork, self).__init__()
        #self.flatten = nn.Flatten()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),            
        )
        #print('SHAPE', self.shape)


        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(53*53*16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        #x = self.flatten(x)
        x = self.features(x)
        #print(x.shape)
        x = self.classifier(x)
        return x



# model = MyNetwork()

# device = "cpu"

# criterion = nn.NLLLoss()
# optimizer = optim.Adam(model.classifier.parameters())

# # Using our model to predict the label
# def predict(image, model, labels):
#     output = model.forward(image)
#     output = torch.exp(output)
    
#     # Get the top predicted class and the output percentage
#     probs, classes = output.topk(1, dim=1)
#     equals = classes == labels.view(*labels.shape)
#     return probs.squeeze().item(), classes.squeeze().item(), equals

# directory = '/Users/joannelin/Desktop/Motorola/mTBIplayground/'
# file_path = os.path.join(directory, 'accuracy_epoch.csv')



# def train_model(epochs):
#     train_loss = 0
#     val_loss = 0
#     train_accuracy = 0
#     val_accuracy = 0

#     with open(file_path, 'w', newline='') as f:
#         fieldNames = ['Epoch', 'Train_accuracy','Train_loss', 'Validation_accuracy', \
#             'Validation_loss', 'Test_accuracy','Test_loss']
#         myWriter = csv.DictWriter(f, fieldnames=fieldNames)
#         myWriter.writeheader()
    
#         for epoch in range(epochs):
#             train_loss = 0
#             val_loss = 0
#             train_accuracy = 0
#             val_accuracy = 0

#             # Training the model
#             model.train()
#             counter = 0
#             for inputs, labels in train_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 optimizer.zero_grad()
#                 output = model.forward(inputs)
#                 loss = criterion(output, labels)
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item()*inputs.size(0)
                
#                 ps = torch.exp(output)
#                 top_p, top_class = ps.topk(1, dim=1)
#                 equals = top_class == labels.view(*top_class.shape)
#                 train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
#                 # Print the progress
#                 counter += 1
#                 print(counter, "/", len(train_loader))
                
#             # Evaluating the model
#             model.eval()
#             # Tell torch not to calculate gradients
#             with torch.no_grad():
#                 counter = 0
#                 for inputs, labels in val_loader:
#                     inputs, labels = inputs.to(device), labels.to(device)
#                     output = model.forward(inputs)
#                     valloss = criterion(output, labels)
#                     val_loss += valloss.item()*inputs.size(0)
                    
#                     output = torch.exp(output)
#                     top_p, top_class = output.topk(1, dim=1)
#                     equals = top_class == labels.view(*top_class.shape)
#                     val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
#             # Get the average loss for the entire epoch
#             train_loss = train_loss/len(train_loader.dataset)
#             val_loss = val_loss/len(val_loader.dataset)
#             # Print out the information

#             print('Train Accuracy: ', train_accuracy/len(train_loader))
#             print('Validation Accuracy: ', val_accuracy/len(val_loader))
#             print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, val_loss))

#             model.eval()

#             test_set = datasets.ImageFolder("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/yes_no_split/test", transform = transformations)
#             test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

#             count_true = 0
#             y_pred = []
#             y_true = []
#             test_accuracy = 0
#             test_loss = 0

#             for inputs, labels in test_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 pred_prob, pred_class, equals= predict(inputs, model, labels)

#                 # loss = criterion(output, labels)
#                 # test_loss += loss.item()*inputs.size(0)

#                 y_pred.extend([pred_class]) # Save Prediction
#                 y_true.extend(labels) # Save Truth

#                 if equals == True:
#                     count_true+=1
#                 # test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

#             # Compute the average test loss and test accuracy
#             test_loss = test_loss / len(test_loader.dataset)
#             test_accuracy = count_true / len(test_loader.dataset)

#             print('Test Accuracy: ', test_accuracy)
#             print('Epoch: {} \tTesting Loss: {:.6f}'.format(epoch, test_loss))

#             c_matrix = confusion_matrix(y_true, y_pred)
#             plt.figure(figsize = (2, 2))
#             print(c_matrix)

#             print(epoch, train_accuracy, train_loss, val_accuracy, val_loss, test_accuracy, test_loss)

#             myWriter.writerow({'Epoch' : epoch+1, 'Train_accuracy' : train_accuracy/len(train_loader), \
#                      'Train_loss' : train_loss, 'Validation_accuracy' : val_accuracy/len(val_loader), \
#                      'Validation_loss' : val_loss, 'Test_accuracy' : test_accuracy,'Test_loss' : test_loss})

#             # with open('accuracy_epoch.csv', mode='a', newline='') as file:
#             #     writer = csv.writer(file)
#             #     writer.writerow({'Epoch' : epoch+1, 'Train_accuracy' : train_accuracy, \
#             #         'Train_loss' : train_loss, 'Validation_accuracy' : val_accuracy, \
#             #         'Validation_loss' : val_loss, 'Test_accuracy' : test_accuracy,'Test_loss' : test_loss})


# train_model(30)
# print(type(model))
# torch.save(model.state_dict().copy(), "/Users/joannelin/Desktop/Motorola/mTBIplayground/binary_classification_model.pth")


# test_set = datasets.ImageFolder("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/yes_no_split/test", transform = transformations)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

# # y_true = []
# # y_pred = []
# # count_true = 0

# # for inputs, labels in test_loader:
# #     inputs, labels = inputs.to(device), labels.to(device)
# #     pred_prob, pred_class, equals, output = predict(inputs, model)

# #     output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
# #     y_pred.extend(output) # Save Prediction

# #     labels = labels.data.cpu().numpy()
# #     y_true.extend(labels) # Save Truth

# #     # print("The model is ", pred_prob*100, "% certain that the image has a predicted class of", pred_class)
# #     if equals.squeeze().item() == True:
# #        count_true+=1
# #     # print("Answer is:", equals.squeeze().item()) 


# # c_matrix = confusion_matrix(y_true, y_pred)
# # print(c_matrix)
