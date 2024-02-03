import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

learning_rate = 0.01
batch_size = 64
n_epochs = 30  
num_features = 28 * 28
num_classes = 10

train_all = datasets.MNIST ('../data', train =True , download = True, transform = transforms.ToTensor() ) # 60K images
train_data , val_data = torch . utils . data . random_split (
    train_all , [50000 , 10000 ], torch.Generator(). manual_seed(0)) # train : 50K ; val : 10K
test_data = datasets.MNIST ('../data', train = False, transform = transforms.ToTensor() ) # test : 10K

train_loader = DataLoader(dataset = train_data, batch_size = 64, shuffle = True)
val_loader = DataLoader(dataset = val_data, batch_size = 64, shuffle = True)
test_loader = DataLoader(dataset = test_data, batch_size = 64, shuffle = False)
for (data, target) in train_loader:
    print('data:', data.size(), 'type:', data.type())
    print('target:', target.size(), 'type:', target.type())
    break

class Softmax(torch.nn.Module):
    def __init__(self):
        super(Softmax,self).__init__()
        self.linear = torch.nn.Linear(num_features,num_classes)  # softmax function

    def forward(self, x):
        x = x.view(-1,num_features)
        x = self.linear(x)
        return x

def train_model(model, train_loader, criterion, optimizer):
    for features, targets in train_loader:
        model.train()
        # forward pass
        output = model(features)
        # Calculate the loss
        loss = criterion(output, targets)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        # update model parameters
        optimizer.step()
      
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss, total_correct = 0,0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            output = model(x_test)
            test_loss += criterion(output, y_test).item()
            # count number of correct
            pred = output.argmax(dim =1,keepdim=True)
            total_correct += pred.eq(y_test.view_as(pred)).sum().item()
    
    accuracy = 100. * total_correct/ len(test_loader.dataset)
    print(f'Test Accuracy is {accuracy}. ')
    return accuracy

class OnelayerClassifier(torch.nn.Module):
    def __init__(self):
        super(OnelayerClassifier,self).__init__()
        self.fc1 = torch.nn.Linear(num_features,1024)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(1024,num_classes)  # softmax function

    def forward(self, x):
        x = x.view(-1,num_features)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class MLP(torch.nn.Module):
    def __init__(self, L, hidden_units):
        super(MLP,self).__init__()
        # input layers
        layers = [torch.nn.Linear(num_features,hidden_units),torch.nn.ReLU()]
        # hidden layers
        for _ in range(1,L):
            layers += [torch.nn.Linear(hidden_units,hidden_units),torch.nn.ReLU()]
        # output layers
        layers += [torch.nn.Linear(hidden_units,num_classes)] 
        self.network = torch.nn.Sequential(*layers)


    def forward(self, x):
        x = x.view(-1, num_features)
        return self.network(x)

def calculate_hidden_units(L):
    if L == 1:
        return 1024
    else:
        m = (-L-794+ np.sqrt(L**2 + 3257908*L - 2625884))/ (2*(L-1))
        return int(m)


def main(q):
    if q == 1:
        print('Answers for question 1:')
        # initialize
        model_Softmax = Softmax()
        print(model_Softmax)

        # optimize and cost
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_Softmax.parameters(), lr= learning_rate)

        # train and test
        train_model(model_Softmax, train_loader, criterion, optimizer)
        test_model(model_Softmax, test_loader, criterion)
    if q == 2:
        print('Answers for question 2:')
        # initialize
        model_OneLayerClassifier = OnelayerClassifier()
        print(model_OneLayerClassifier)

        # optimize and cost
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_OneLayerClassifier.parameters(), lr= learning_rate)

        # train and test
        train_model(model_OneLayerClassifier, train_loader, criterion, optimizer)
        test_model(model_OneLayerClassifier, test_loader, criterion)
    if q == 4:
        print('Answers for question 2:')
        accuracies = []
        for l in range(1,9):
            m = calculate_hidden_units(l)
            model_mlp = MLP(l,m)
            print(model_mlp)

            # optimize and cost
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.SGD(model_mlp.parameters(), lr= learning_rate)

            # train and test
            train_model(model_mlp, train_loader, criterion, optimizer)
            accuracy = test_model(model_mlp, test_loader, criterion)
            accuracies.append(accuracy)
            print(f'For L hidden layers={l}, hiddern units: ={m}')
        print('The accuracy list is ',accuracies)

        plt.plot(range(1,9),accuracies, marker ='o')
        plt.xlabel('Number of Hidden Layers')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracies for Different Number of Hidden Layers')
        plt.xticks(range(1,9))
        plt.savefig('MLP accuracies.png')
        plt.show()


    
if __name__=='__main__':
    main(4)

