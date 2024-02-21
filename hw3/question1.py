import os, torch, shutil, numpy as np
from glob import glob; from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
torch.manual_seed(2024)

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  print("We don't have GPU")
  device = torch.device('cpu')

def plot_accuracy_loss(training_results):
  plt.subplot(2,1,1)
  plt.plot(training_results['training loss'],'r', label='traning loss')
  plt.ylabel('loss')
  plt.title('training loss iterations')
  plt.subplot(2,1,2)
  plt.plot(training_results['validation accuracy'], label='validation accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('iterations')
  plt.legend()
  plt.savefig('question1.png')
  plt.show()

"""
# If we use mean and std in the training set, the code should be as followed
def calculate_mean_std(loader):
    means = []
    stds = []
    
    with torch.no_grad():
        for images, _ in loader:
            # Compute mean and std for each batch
            batch_mean = torch.mean(images, dim=[0, 2, 3])
            batch_std = torch.std(images, dim=[0, 2, 3])
            
            means.append(batch_mean)
            stds.append(batch_std)
    
    # Convert lists to tensors
    means = torch.stack(means)
    stds = torch.stack(stds)
    
    # Compute the mean and std across all batches
    mean = torch.mean(means, dim=0)
    std = torch.mean(stds, dim=0)

    return mean, std
"""
mean, std = [0.4686, 0.4686, 0.4686], [0.2775,0.2775, 0.2775]
"""
class CustomDataset(Dataset):
    
    def __init__(self, root, csv_file, dataset, transform = None):

      # Image directory
      self.root = root
      
      # The transform is going to be used on image
      self.transform = transform
      data_dircsv_file= os.path.join(self.root, csv_file)
      df = pd.read_csv(data_dircsv_file)

      #load the csv file with different dataset
      # column0: class id; column1:filepaths; column2:labels; column3: data set
      self.data_name = df.loc[df["data set"]==dataset]

      self.len = self.data_name.shape[0]
  
    def __len__(self): 
      return self.len

    def __getitem__(self, idx):
        
        # image file path
        img_name = os.path.join(self.root, self.data_name.iloc[idx,1])
        # open image file
        image = Image.open(img_name)

        # The class label for the image
        y = self.data_name.iloc[idx,0]
        # Convert label to tensor

        if self.transform:
          image = self.transform(image)
        
        return image,y # return a tuple of (image, y_label's id)

csv_file = 'sports.csv'
root = "/content/gdrive/MyDrive/Colab Notebooks/hw3/data"
im_size = 224
best_val_accuracy = 0.0
best_model_path = "best_model.pth" 
num_epochs = 5


#most images seem to have 3 channels (presumably RGB images), at least one image appears to have just 1 channel (a grayscale image).
previous_transform = transforms.Compose([transforms.Resize((im_size, im_size)),transforms.Grayscale(num_output_channels=3),transforms.ToTensor()])
previous_transform_dataset = CustomDataset(root = root, csv_file=csv_file, dataset = "train", transform = previous_transform)
previous_transform_dataloader = DataLoader(previous_transform_dataset, batch_size = 64, shuffle = True) 
mean, std = calculate_mean_std(previous_transform_dataloader)
print(f'Without using any published weights, the mean and std for nomalization are: {mean} and {std} respectively')


# use a random mean and std
tfs = transforms.Compose([transforms.Resize((im_size, im_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(), 
            transforms.Normalize(mean = mean, std = std)])
   
# get dataset
train_dataset = CustomDataset(root = root, csv_file=csv_file, dataset = "train", transform = tfs)
validation_dataset = CustomDataset(root = root,csv_file=csv_file, dataset = "valid", transform = tfs)
test_dataset = CustomDataset(root = root,csv_file=csv_file, dataset = "test", transform = tfs)
    
# get dataloader
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True) 
validation_loader = DataLoader(validation_dataset, batch_size = 64, shuffle = False) 
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

#print("length of training set: ", len(train_dataset))
#print("length of validation set:", len(validation_dataset))
#print("length of test set: ", len(test_dataset))

#print("Shape of the first sample: ",train_dataset[0][0].shape)
#print("Label for the first data: ",train_dataset[0][1])

class AlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        
        # Adjusting the first convolutional layer to use padding=0
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=0),  # No padding here
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Traning loop
def train(model,criterion, train_loader, validation_loader,optimizer, num_epochs, best_val_accuracy):
  useful_stuff ={'training loss':[],'validation accuracy':[]}
  for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model(inputs)

      optimizer.zero_grad()
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      # loss for each iterations
      useful_stuff['training loss'].append(loss.data.item())
    
    correct = 0
    model.eval()
    with torch.no_grad():
      for x,y in validation_loader:
        x,y = x.to(device), y.to(device)
        yhat = model(x)
        _, label = torch.max(yhat.data,1)
        correct += (label == y).sum().item()

    # accuracy for each epoch
    accuracy = 100 *(correct/len(validation_dataset))
    useful_stuff['validation accuracy'].append(accuracy)

    # save the best model
    if accuracy > best_val_accuracy:
      best_val_accuracy = accuracy
      torch.save(model.state_dict(),best_model_path)
      print(f"Update Best Model with Validation Accuracy: {best_val_accuracy:.2f}%")
  return useful_stuff


model = AlexNet(num_classes = 100).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr =0.0001)

train_result = train(model,criterion, train_loader, validation_loader,optimizer, num_epochs, best_val_accuracy)

plot_accuracy_loss(train_result)
print("The accuracy for validation set: \n", train_result['validation accuracy'])

# reify the form from your baize
model.load_state_dict(torch.load(best_model_path))

# Test accuracy:
correct = 0
model.eval()
with torch.no_grad():
  for test_x, test_y in test_loader:
    test_x, test_y = test_x.to(device), test_y.to(device)
    test_yhat = model(test_x)
    _, predicted = torch.max(test_yhat, 1)
    correct += (predicted == test_y).sum().item()
print(f'Accuracy for the test dataset is: {100*correct/len(test_dataset):.2f}%')

