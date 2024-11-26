import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, f1_score, confusion_matrix, accuracy_score,classification_report
import torchvision
import torch.nn as nn
import pickle

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

trainset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
class_labels = trainset.classes

model = nn.Sequential(
    nn.Linear(50,512),
    nn.ReLU(),
    nn.Linear(512,1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Linear(1024,10),
                      )


# Load info
model.load_state_dict(torch.load("Models/MLP/Hidden_Layer/mlp_hl_model.pth"))
model.eval()
print("Model loaded successfully.")

# Load the feature vectors and labels
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

feature_vectors_test_50 = data['feature_vectors_test_50']
labels_test = torch.tensor(data['labels_test'])


with torch.no_grad():
    Xtest = torch.tensor(feature_vectors_test_50)
    outputs = model(Xtest)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = accuracy_score(labels_test, predicted)
    recall = recall_score(labels_test, predicted, average='weighted')
    f1 = f1_score(labels_test, predicted, average='weighted')
    cm = confusion_matrix(labels_test, predicted)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Plt
    plt.figure(figsize=(15, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    # plt.savefig("confusion_matrix_mlp_hl.png", dpi=300, bbox_inches='tight') 

    plt.show()
    print("Classification report: ",classification_report(labels_test, predicted))
