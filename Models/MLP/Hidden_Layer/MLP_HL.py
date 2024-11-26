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
import seaborn as sns
from sklearn.metrics import recall_score,f1_score,confusion_matrix,classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pickle

# transformation required for normalization and resizing
transform = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data',train = False, download=True,transform = transform)

with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)


feature_vector_50 = data['feature_vector_50']
feature_vectors_test_50 = data['feature_vectors_test_50']
labels = data['labels']  
labels_test = data['labels_test']

# x_train = feature_vector_50
# y_train = labels
# x_test = feature_vectors_test_50
# y_test = labels_test

# define model
model = nn.Sequential(
    nn.Linear(50,512),
    nn.ReLU(),
    nn.Linear(512,1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Linear(1024,10),
                      )
print(model)
# loss function is Cross Entropy
loss_fn = nn.CrossEntropyLoss()

# optimizer is SGD and momentum is 0.9
# thinking of changing up the learning rate, maybe itll give better results
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
Xtensor = torch.tensor(feature_vector_50)
labels = torch.tensor(labels)
for n in range(1000):
  model.train()
  # predict, get loss,
  prediction = model(Xtensor)
  loss = loss_fn(prediction,labels)
  # zero the gradient
  optimizer.zero_grad()
  # gradient descent
  loss.backward()
  optimizer.step()

  print(f"Epoch {n+1}, Loss: {loss.item()}")
  model.eval()
  correct = 0
  total = 0
# disable gradient
  with torch.no_grad():
    Xtest = torch.tensor(feature_vectors_test_50)
    labels_test = torch.tensor(labels_test)
    outputs = model(Xtest)
    _, predicted = torch.max(outputs.data, 1)
    total += labels_test.size(0)
    correct += (predicted == labels_test).sum().item()
    print(f'Accuracy of the network on the test images: {100 * correct / total} %')
torch.save(model.state_dict(), "mlp_hl_model.pth")



