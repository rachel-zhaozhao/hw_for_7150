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
  plt.savefig('hw3/question2.png')
  plt.show()


mean, std = [0.485, 0.456, 0.406], [0.229,0.224, 0.225]

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
best_model_path = "hw3/best_model2.pth" 
num_epochs = 5
num_classes = 100

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
    
    # compute accuracy for validation 
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

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
model.to(device)


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

