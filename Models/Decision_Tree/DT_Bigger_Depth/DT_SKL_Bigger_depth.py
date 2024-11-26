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
from sklearn.metrics import recall_score,f1_score,accuracy_score,confusion_matrix,classification_report
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

# Extract data from the dictionary
feature_vector_50 = data['feature_vector_50']
feature_vectors_test_50 = data['feature_vectors_test_50']
labels = data['labels'] 
labels_test = data['labels_test']


# decision tree with sklearn
DTC = DecisionTreeClassifier(criterion ="gini",max_depth=70)
DTC=DTC.fit(feature_vector_50,labels)

prediction = DTC.predict(feature_vectors_test_50)



# since the results are near perfect for DT i didnt bother using the same testset for all ( as i did with CNN)

cm = confusion_matrix(labels_test, prediction)

print("Classification report: ",classification_report(labels_test, prediction))

plt.figure(figsize=(15, 5))
class_labels = trainset.classes
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.savefig("confusion_matrix_DT_biggers.png", dpi=300, bbox_inches='tight')

plt.show()
plt.close()

