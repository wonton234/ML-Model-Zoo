import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix


with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

# get predifined data
feature_vector_50 = data['feature_vector_50']
feature_vectors_test_50 = data['feature_vectors_test_50']
labels = data['labels'] 
labels_test = data['labels_test']


transform = transforms.Compose(
    # resize images and normalize
    [transforms.Resize((224,224)),
     transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data',train = False, download=True,transform = transform)


# torch was only used before this point not after
class GNB:
  x_train:np.ndarray
  y_train:np.ndarray
  # initialize the x
  def __init__(self, x_train, y_train):
    self.x_train = x_train
    self.y_train = y_train

  def fit(self):
    # get all classes
    self.unique_classes = np.unique(self.y_train)
    self.params = []

    # compute the mean and variance of the features in each class
    # since this is a multi-dim array, we apply the mean and variance to each column
    for c in self.unique_classes:
      x_train_c = self.x_train[self.y_train == c]
      self.params.append({
        'mean': x_train_c.mean(axis = 0),
        'var': x_train_c.var(axis = 0)
      })

# predict the class for the feature vectors provided
  def predict(self, x_test):
    y_pred = [self._predict(x) for x in x_test]
    return np.array(y_pred)


  def _predict(self, x):
    posteriors = []
    for i, c in enumerate(self.unique_classes):
      # take out mean and variance and calculate likelihood
      mean = self.params[i]['mean']
      var = self.params[i]['var']

      posterior = np.sum(np.log(self.likelihood(x, mean, var)))
      posteriors.append(posterior)
    return self.unique_classes[np.argmax(posteriors)]

# estimate likelihood
  def likelihood(self, x, mean, var):
    numerator = np.exp(-((x - mean) ** 2) / (2 * var))
    denominator = np.sqrt(2 * np.pi * var)
    return numerator/denominator


# run the fit and predict functions
GNBClassifier = GNB(feature_vector_50,labels)
GNBClassifier.fit()


GND_predictions = GNBClassifier.predict(feature_vectors_test_50)

print("Classification report: ",classification_report(labels_test, GND_predictions))

cm = confusion_matrix(labels_test, GND_predictions)
class_labels = trainset.classes
# Create confusion matrix plot
plt.figure(figsize=(15, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("GNB_Manual")

# Save the confusion matrix plot as an image (used chatgpt for this part)
# plt.savefig("confusion_matrix_GNB_Manual.png", dpi=300, bbox_inches='tight') 

plt.show()
plt.close() 

