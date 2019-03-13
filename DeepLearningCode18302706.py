import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F
from skimage import transform
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

num_epochs = 20 ;
batch_size = 50;
learning_rate = 0.015;


# Creating the CNN Model
class Swish(nn.Module):
  def forward(self, input):
    return (input * torch.sigmoid(input))
  
  def __repr__(self):
    return self.__class__.__name__ + ' ()'

# Creating the CNN Model
class CNNModel(nn.Module):
    def __init__ (self):
        super(CNNModel,self).__init__()
        #CNN Model 1
        self.cnn1=nn.Conv2d(in_channels=1,out_channels=32, kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(32)
        self.swish1=Swish()
        nn.init.xavier_normal(self.cnn1.weight)
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        #CNN Model 2
        self.cnn2=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(64)
        self.swish2=Swish()
        nn.init.xavier_normal(self.cnn2.weight)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(64*7*7,10)
        
        
        
    def forward(self,x):
        out=self.cnn1(x)
        out=self.bn1(out)
        out=self.swish1(out)
        out=self.maxpool1(out)
        out=self.cnn2(out)
        out=self.bn2(out)
        out=self.swish2(out)
        out=self.maxpool2(out)

        out=out.view(out.size(0),-1)
        out=self.fc1(out)
        
        return F.log_softmax(out, dim=1)

# Plotting the error and loss graphs    
def plot_graph(train_x, train_y, test_x, test_y, ylabel=''):
    fig = plt.figure()
    plt.plot(train_x, train_y, color='blue')
    plt.plot(test_x, test_y, color='red')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()
    
 #Training the Model   
def train(model, device, train_loader, optimizer, epoch, losses=[], counter=[], errors=[]):
    model.train()
    correct=0  
    criterion = nn.CrossEntropyLoss();

    for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            images = Variable(images.float())
            labels = Variable(labels)
            output = model(images)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            #losses.append(loss.data[0]);
        
            if (i+1) % 100 == 0:
                print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_loader)//batch_size, loss.data[0]))
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            losses.append(loss.item())
            counter.append((i*batch_size) + ((epoch-1)*len(train_loader.dataset)))
    errors.append(100. * (1 - correct / len(train_loader.dataset))) 

#Testing the Model    
def test(model, device, test_loader, losses=[], errors=[]):    
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = Variable(images.float())
            output = model(images)
            test_loss += F.nll_loss(output, labels , reduction='sum').item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)
            
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))    
    losses.append(test_loss)
    errors.append(100. *  (1 - correct / len(test_loader.dataset)))    
    print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100 * correct / total))    
    
#Saving the Predictions
def save_predictions(model, device, test_loader, path):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = F.softmax(model(data), dim=1)
            with open(path, "a") as out_file:
                np.savetxt(out_file, output)

#Main Function
def main():
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # data transformation
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
#Importing the Dataset
    train_dataset = dsets.FashionMNIST(root='./data', 
                            train=True, 
                            transform=transform,
                            download=True
                           )

    test_dataset = dsets.FashionMNIST(root='./data', 
                            train=False, 
                            transform=transform
                           )

    # data loaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size,            
                                           shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False, **kwargs)

# instance of the Conv Net

    model = CNNModel().to(device);

#scheduler and optimizer

    
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate);
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.003)

    # lists for saving history
    train_losses = []
    train_counter = []
    test_losses = []
    train_errors = []
    test_errors = []
    test_counter = [i*len(train_loader.dataset) for i in range(num_epochs)]
    #print(test_counter)

    error_counter = [i*len(train_loader.dataset) for i in range(num_epochs)]
    #print(error_counter)
    # global training and testing loop
    for epoch in range(0, num_epochs):
        
        train(model, device, train_loader, optimizer, epoch, losses=train_losses, counter=train_counter, errors=train_errors)
        test(model, device, test_loader, losses=test_losses, errors=test_errors)
		
    # plotting training history
    
    print(error_counter)
    print(test_losses)
    plot_graph(train_counter, train_losses, test_counter, test_losses, ylabel='negative log likelihood loss')
    plot_graph(error_counter, train_errors, error_counter, test_errors, ylabel='error (%)')
    
# Save model
    save_model = True
    if save_model is True:
    #saves only params
       torch.save(model.state_dict(), 'model_18302706.pt')
       model.load_state_dict(torch.load('model_18302706.pt'))
    save_predictions(model, device, test_loader, 'predictions_18302706.txt')
if __name__ == '__main__':
    main()
