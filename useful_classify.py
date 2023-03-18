from cv2 import transform
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import cv2
import os


# root_dir = "/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/yes_no_split"

# for image in os.listdir(root_dir+"/train/yes"):
#     img = cv2.imread(os.path.join(root_dir,"train", "yes", image))
#     print(img.shape)

# for image in os.listdir(root_dir+"/train/no"):
#     img = cv2.imread(os.path.join(root_dir,"train", "no", image))
#     print(img.shape)

transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# for image in os.listdir(root_dir+"/train/yes"):
#     img = cv2.imread(os.path.join(root_dir,"train", "yes", image))
#     img_normalized = transformations(img)
#     print(img_normalized.shape)



# Define the dataset
train_set = datasets.ImageFolder("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/yes_no_split/train", transform = transformations)
val_set = datasets.ImageFolder("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/yes_no_split/val", transform = transformations)






# Define the batch size
batch_size = 6
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size =batch_size, shuffle=True)


    

class MyNetwork(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Get pretrained model using torchvision.models as models library
model = MyNetwork()
# model = models.densenet161(pretrained=True)
# Turn off training for their parameters
for param in model.parameters():
    param.requires_grad = False

# Create new classifier for model using torch.nn as nn library
classifier_input = model.classifier
num_labels = 2
classifier = nn.Sequential(nn.Flatten(),
                           nn.Linear(44944, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))
model.classifier = classifier


# Find available device
device = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Set the error function using torch.nn as nn library
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters())


epochs = 5
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0
    
    # Training the model
    model.train()
    counter = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*inputs.size(0)
        
        # Print the progress
        counter += 1
        print(counter, "/", len(train_loader))
        
    # Evaluating the model
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            valloss = criterion(output, labels)
            val_loss += valloss.item()*inputs.size(0)
            
            output = torch.exp(output)
            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Print the progress of our evaluation
            counter += 1
    
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(val_loader.dataset)
    # Print out the information
    print('Accuracy: ', accuracy/len(val_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

model.eval()


# def process_image(image_path):
#     # Load Image
#     img = Image.open(image_path).convert('L')
    
#     # Get the dimensions of the image
#     width, height = img.size
#     img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
#     width, height = img.size
    
#     # Set the coordinates to do a center crop of 224 x 224
#     left = (width - 224)/2
#     top = (height - 224)/2
#     right = (width + 224)/2
#     bottom = (height + 224)/2
#     img = img.crop((left, top, right, bottom))
    
#     img = np.array(img)
#     img = np.expand_dims(img, axis=2)
#     img = img/255
#     img = (img - 0.5)/0.5
#     img = img[np.newaxis,:]
    
#     # Turn into a torch tensor
#     image = torch.from_numpy(img)
#     image = image.float()
#     return image

# Using our model to predict the label
def predict(image, model):
    print(image.shape)
    output = model.forward(image)
    output = torch.exp(output)
    
    # Get the top predicted class and the output percentage
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()

# def show_image(image):
#     image = image.numpy()
    
#     # Unnormalize
#     image = image.squeeze()
#     image = image * 0.226 + 0.445
    
#     fig = plt.figure(figsize=(25, 4))
#     plt.imshow(image, cmap='gray')

# print()
image = transforms("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/mTBI_data_new_2022/mTBI_Data_JPEGs/D003/T1/MRI BRAIN_BRAIN STEM W_O CONT_5115750/export--73327752.jpg")
pred_prob, pred_class = predict(image, model)
# show_image(image)
print("The model is ", pred_prob*100, "% certain that the image has a predicted class of ", pred_class)