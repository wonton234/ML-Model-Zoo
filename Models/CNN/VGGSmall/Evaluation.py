import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import seaborn as sns

from sklearn.metrics import classification_report,confusion_matrix

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
          nn.MaxPool2d(2,2),

          nn.Conv2d(128,256,3,1,1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.MaxPool2d(2,2),

          

          nn.Conv2d(256,512,3,1,1),
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
model = VGG11(in_channels=1, num_classes=10)
model.load_state_dict(torch.load("Models/CNN/VGGSmall/VGGSmall.pth",map_location = device))
model.eval() 
if __name__ == '__main__':
   

    transform2 = transforms.Compose(
        # resize images
        # dont randomize image creation because i have created the test set already (testloader_param and testset_idx)
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data',train = False, download=True,transform = transform2)


    testset_indices = torch.load('testset_idx.pth')

    testset2 = Subset(testset, testset_indices)

    testloader_params = torch.load('testloader_params.pth')

    testloader = DataLoader(testset2, **testloader_params)


    correct = 0
    total = 0
    predictions = []
    true_labels = []

    with torch.no_grad():  
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # highest score idx
            _, predicted = torch.max(outputs, 1)  
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f"Accuracy: ", accuracy)
    
    print("Classification Report: ")
    print(classification_report(true_labels, predictions))
    class_labels = testset.classes
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)

    plt.title("VGG Small")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    # plt.savefig("confusion_matrix_VGGSmall.png", dpi=300, bbox_inches='tight')
    plt.show()