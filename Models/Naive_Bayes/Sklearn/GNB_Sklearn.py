import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import recall_score,f1_score,accuracy_score,confusion_matrix,classification_report
import pickle
from sklearn.naive_bayes import GaussianNB


transform = transforms.Compose(
      [transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
      ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data',train = False, download=True,transform = transform)

# Load the processed data
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)


feature_vector_50 = data['feature_vector_50']
feature_vectors_test_50 = data['feature_vectors_test_50']
labels = data['labels']  
labels_test = data['labels_test']

# use the model from sklearn
Gauss = GaussianNB()
Gauss.fit(feature_vector_50,labels)


# Predict class labels for test set
predictions = Gauss.predict(feature_vectors_test_50)
class_labels = trainset.classes
recall = recall_score(labels_test, predictions, average='weighted')
f1 = f1_score(labels_test, predictions, average='weighted')
print("Recall:", recall)
print("F1 Score:", f1)

accuracy = accuracy_score(labels_test, predictions)
cm = confusion_matrix(labels_test, predictions)
print("Classification report: ",classification_report(labels_test, predictions))
plt.figure(figsize=(15, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
print("Accuracy:", accuracy)
plt.show()
# plt.savefig("confusion_matrix_GNB_Sklearn.png", dpi=300, bbox_inches='tight') 
plt.close()