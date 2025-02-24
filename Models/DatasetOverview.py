import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
from sklearn.decomposition import PCA
import pickle

def main():
  # resize and normalize
  transform = transforms.Compose(
      [transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
      ])
  trainset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
  testset = torchvision.datasets.CIFAR10(root='./data',train = False, download=True,transform = transform)


  # subset creator class
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



  trainloader = DataLoader(trainset2, batch_size=8, shuffle=True, num_workers=2)
  testloader = DataLoader(testset2, batch_size=8, shuffle=False, num_workers=2)

  # should be 625 in trainloader and 125 in testloader ( since batch_size is 8)

  print(f"Number of training samples: {len(trainloader.dataset)}")
  print(f"Number of testing samples: {len(testloader.dataset)}")
  print(f"Number of training batches: {len(trainloader)}")
  print(f"Number of testing batches: {len(testloader)}")


  model = models.resnet18(pretrained=True)
  # remove last layer
  model = torch.nn.Sequential(*list(model.children())[:-1])
  model.eval()

  resnet18 = model.to(device)
  feature_vectors = []
  feature_vectors_test = []
  labels = []
  labels_test=[]
  # dont use gradient and have a counter to see how many batches have been processed
  counter = 0 
  counter_test = 0
  with torch.no_grad():
    for inputs,label in trainloader:
      counter+=1
      print(counter)
      inputs = inputs.to(device)
      label = label.to(device)
      labels.append(label)
      # process each batch in resnet18
      outputs = resnet18(inputs)
      outputs=outputs.view(outputs.size(0),-1)
      # append to feature vectors
      feature_vectors.append(outputs)
    for inputs2,labels2 in testloader :
      counter_test+=1
      print(counter_test)
      inputs2 = inputs2.to(device)
      labels2 = labels2.to(device)
      labels_test.append(labels2)  
      outputs=resnet18(inputs2)
      
      outputs=outputs.view(outputs.size(0),-1)
      # process each batch in resnet18
      feature_vectors_test.append(outputs)
  feature_vectors = torch.cat(feature_vectors,dim=0).cpu()
  feature_vectors_test = torch.cat(feature_vectors_test,dim=0).cpu()
  labels = torch.cat(labels, dim=0).cpu()
  labels_test = torch.cat(labels_test, dim=0).cpu()

  print("Shape: ",feature_vectors.shape)
  print("Shape test: ", feature_vectors_test.shape)


  # reduce size of feature vectors
  print("Shape test: ", feature_vectors_test.shape)
  feature_vector_np = feature_vectors.cpu().numpy()
  feature_vector_test_np = feature_vectors_test.cpu().numpy()
  pca = PCA(n_components=50)
  feature_vector_50=pca.fit_transform(feature_vector_np)
  feature_vectors_test_50 = pca.transform(feature_vector_test_np)
  print(sum(pca.explained_variance_ratio_))
  print("Shape: ",feature_vector_50.shape)
  print("Shape test: ", feature_vectors_test_50.shape)
  
  # Save the data ( this last part has been generated by chatgpt to dump all the feature vectors and
  # what not into a file so i can use it later)
  data = {
    "feature_vector_50": feature_vector_50,
    "feature_vectors_test_50": feature_vectors_test_50,
    "labels": labels.numpy(),
    "labels_test": labels_test.numpy()
    }

  with open("processed_data.pkl", "wb") as f:
    pickle.dump(data, f)
  print("Data saved to 'processed_data.pkl'")

if __name__ == '__main__':
  main()