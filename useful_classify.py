import intensity_normalize
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
import os
from torchvision.models.vgg import vgg16, VGG16_Weights


# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# Define the dataset
# dataset_train = datasets.ImageFolder("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/yes_no_split", transform=transformations)
# dataset_val = datasets.ImageFolder("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/yes_no_split", transform=transformations)

train_set = datasets.ImageFolder("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/yes_no_split/train", transform = transformations)
val_set = datasets.ImageFolder("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/yes_no_split/validate", transform = transformations)

# Calculate the split sizes
# train_size_1 = int(0.7 * len(dataset_yes))
# val_size_1 = len(dataset_yes) - train_size_1

# train_size_2 = int(0.7 * len(dataset_no))
# val_size_2 = len(dataset_no) - train_size_2

# # Split the dataset into train and validation sets
# train_set_1, val_set_1 = random_split(dataset_yes, [train_size_1, val_size_1])
# train_set_2, val_set_2 = random_split(dataset_no, [train_size_2, val_size_2])

# Define the batch size
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size =batch_size, shuffle=True)

# train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_set_1, num_workers=4)
# train_loader2 = torch.utils.data.DataLoader(dataset_no, batch_size=batch_size, sampler=train_set_2, num_workers=4)
# train_loader = torch.utils.data.ConcatDataset([train_loader, train_loader2])

# val_loader = torch.utils.data.DataLoader(dataset_yes, batch_size=batch_size, sampler=val_set_1, num_workers=4)
# val_loader2 = torch.utils.data.DataLoader(dataset_no, batch_size=batch_size, sampler=val_set_2, num_workers=4)
# val_loader = torch.utils.data.ConcatDataset([val_loader, val_loader2])



# Get pretrained model using torchvision.models as models library
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
# Turn off training for their parameters
for param in model.parameters():
    param.requires_grad = False

# Create new classifier for model using torch.nn as nn library
classifier_input = model.classifier[6].in_features
num_labels = 2
classifier = nn.Sequential(
    nn.Linear(classifier_input, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, num_labels),
    nn.LogSoftmax(dim=1)
)
model.classifier = classifier


# Find available device
device = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Set the error function using torch.nn as nn library
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters())


epochs = 10
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
            print(counter, "/", len(val_loader))
    
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(val_loader.dataset)
    # Print out the information
    print('Accuracy: ', accuracy/len(val_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

model.eval()


def process_image(image_path):
    # Load Image
    img = Image.open(image_path).convert('L')
    
    # Get the dimensions of the image
    width, height = img.size
    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    width, height = img.size
    
    # Set the coordinates to do a center crop of 224 x 224
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    
    img = np.array(img)
    img = np.expand_dims(img, axis=2)
    img = img/255
    img = (img - 0.5)/0.5
    img = img[np.newaxis,:]
    
    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image

# Using our model to predict the label
def predict(image, model):
    output = model.forward(image)
    output = torch.exp(output)
    
    # Get the top predicted class and the output percentage
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()

def show_image(image):
    image = image.numpy()
    
    # Unnormalize
    image = image.squeeze()
    image = image * 0.226 + 0.445
    
    fig = plt.figure(figsize=(25, 4))
    plt.imshow(image, cmap='gray')


image = process_image("/Users/joannelin/Desktop/Motorola/mTBIplayground/Datasets/mTBI_data_new_2022/mTBI_Data_JPEGs/D003/T1/MRI BRAIN_BRAIN STEM W_O CONT_5115750/export--73327752.jpg")
top_prob, top_class = predict(image, model)
show_image(image)
print("The model is ", top_prob*100, "% certain that the image has a predicted class of ", top_class)