import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report,recall_score,f1_score,confusion_matrix


if __name__ == "__main__":
  transform2 = transforms.Compose(
      # resize images and randomization ( dont do that for the evaluation)
      [
          transforms.Grayscale(num_output_channels=1),
          transforms.RandomRotation(10),
          transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5))
      ])

  trainset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform2)
  testset = torchvision.datasets.CIFAR10(root='./data',train = False, download=True,transform = transform2)


  def subset_per_class(dataset, num_samples):
      indices = []
      targets = dataset.targets
      for i in range(10):
          class_indices = [idx for idx, target in enumerate(targets) if target == i]
          indices.extend(class_indices[:num_samples])
      return torch.utils.data.Subset(dataset, indices)

  # load into trainloader, testloader

  trainset2 = subset_per_class(trainset,500)
  testset2 = subset_per_class(testset,100)



  trainloader = DataLoader(trainset2, batch_size=16, shuffle=True, num_workers=2)
  testloader = DataLoader(testset2, batch_size=32, shuffle=True, num_workers=2)

  images,labels = next(iter(trainloader))




  class VGG11(nn.Module):
    def __init__(self,in_channels,num_classes=10):
      super(VGG11,self).__init__()
      self.in_channels = in_channels
      self.num_classes = num_classes
    
      self.conv_layers = nn.Sequential(
          nn.Conv2d(1,64,3,1,1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.MaxPool2d(2,2),

          nn.Conv2d(64,128,3,1,1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          
          nn.Conv2d(128,128,3,1,1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          


          nn.Conv2d(128,256,3,1,1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.MaxPool2d(2,2),

          nn.Conv2d(256,256,3,1,1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          

          nn.Conv2d(256,512,3,1,1),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.MaxPool2d(2,2),

          
          nn.Conv2d(512,512,3,1,1),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          


          nn.Conv2d(512,512,3,1,1),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.MaxPool2d(2,2),



          nn.Conv2d(512,512,3,1,1),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.MaxPool2d(2,2),

    

      )

  # fully connected layer
      self.linear_layers = nn.Sequential(
          nn.Linear(512,4096),
          nn.ReLU(),
          nn.Dropout(0.5),

          nn.Linear(4096,4096),
          nn.ReLU(),
          nn.Dropout(0.5),

          nn.Linear(4096,10)
      )

      # forward pass
    def forward(self,x):
      x = self.conv_layers(x)
      x = x.view(x.size(0),-1)
      x = self.linear_layers(x)
      return x
    

  model = VGG11(in_channels=1,num_classes = 10)
  model = model.to(device)


  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  # dynamic learning rate
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

  print(len(trainloader))
  counter = 0
  for epoch in range(15):
    running_loss = 0.0
    model.train()
    for inputs,labels in trainloader:
      counter+=1
      print(counter)
      inputs = inputs.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      # get the output for the image
      outputs = model(inputs)
      # check how close the answer is to the prediction
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for test_inputs, test_labels in testloader:
            test_inputs = test_inputs.to(device)
            test_labels =  test_labels.to(device)
            test_outputs = model(test_inputs)
            test_loss += criterion(test_outputs, test_labels).item()
            _, predicted = test_outputs.max(1)
            correct += (predicted == test_labels).sum().item()
            total += test_labels.size(0)
    print(f"Test Loss: {test_loss / len(testloader)}, Accuracy: {100 * correct / total}%")


  print("Finished")
