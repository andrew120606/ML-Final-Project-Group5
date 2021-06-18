#%%#-----------------------import packages--------------------------------------------------

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, log_loss, accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#%%#-----------------------find the path for all files------------------------------

directory = 'C:/Andrew120606/GWU_DATA_SCIENCE/courses/Machine Learning I-Amir/ML final project/BrainCancerData'
File=[]
for file in os.listdir(directory):
    File+=[file]
print(File)

#%%#-----------------------Data vasulization----------------------------------

directory = "C:/Andrew120606/GWU_DATA_SCIENCE/courses/Machine Learning I-Amir/ML final project/BrainCancerData"
image_names = ["/no_tumor/image(188).jpg","/no_tumor/image(127).jpg","/glioma_tumor/gg (7).jpg","/glioma_tumor/gg (53).jpg", "/meningioma_tumor/m (139).jpg", "/meningioma_tumor/m2 (57).jpg",
              "/pituitary_tumor/p (635).jpg","/pituitary_tumor/p (158).jpg"]

for i in range(len(image_names)):
    ax = plt.subplot(2, 4, i + 1)
    image = load_img(directory + image_names[i], color_mode='rgb', target_size=(239, 239))
    plt.imshow(image)
    plt.title(image_names[i].split('/')[1])
    plt.axis("off")
plt.tight_layout()
plt.savefig('examples.png')
plt.show()

#%%#-----------------------read image and transfer them to array and label them with 0,1,2,3------------------

dataset=[]
labels=[]
mapping={'no_tumor':0, 'pituitary_tumor':1, 'meningioma_tumor':2, 'glioma_tumor':3}

for file in os.listdir(directory):
    path = os.path.join(directory,file)
    for im in os.listdir(path):
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(128,128))
        image=img_to_array(image)
        image=image/255.0
        dataset.append(image)
        labels.append(mapping[file])
print(len(dataset), len(labels))

#%%#---------------------------define X, y-----------------------------------------------------

X = np.array(dataset)
y = np.array(labels)
print(X.shape)
print(y.shape)

#%%#---------------------------define X, y------------------------------------------------------

# labels1 = to_categorical(labels)  # array to matrix
# X = np.array(dataset)
# y = np.array(labels1)
# print(X.shape)
# print(y.shape)

#%%#------------------------data split----------------------------------------------------------

X, y = shuffle(X, y, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

#%%#------------------------3D to 2D----------------------------------------------------------------

X_train_flatten = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
X_test_flatten = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3])
print("X_train_flatten",X_train_flatten.shape)
print("X_test_flatten",X_test_flatten.shape)

#%%#------------------------classifier--------------------------------------------------------

#mlp = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=500, activation='relu',early_stopping=False,momentum=0.9,solver='adam' )
#mlp = MLPClassifier(hidden_layer_sizes=(200,200),max_iter=500, activation='relu',early_stopping=False,momentum=0.9,solver='adam' )
#mlp = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=5000, activation='relu',early_stopping=False,momentum=0.9,solver='adam' )
#mlp = MLPClassifier(hidden_layer_sizes=(20,20,20,20),max_iter=500, activation='relu',early_stopping=False,momentum=0.9,solver='adam' )
#mlp = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=500, activation='logistic',early_stopping=False,momentum=0.9,solver='adam' )
#mlp = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=500, activation='relu',early_stopping=True,momentum=0.9,solver='adam' )
#mlp = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=500, activation='relu',early_stopping=False,momentum=0.9,solver='sgd' )
mlp = MLPClassifier(hidden_layer_sizes=(200,200,20,20,20,20),max_iter=500, activation='relu',early_stopping=False,momentum=0.9,solver='adam' )

#%%#------------------------training and prediction----------------------------------------

mlp.fit(X_train_flatten, y_train)
y_pred = mlp.predict(X_test_flatten)

#%#------------------------evaluation and loss-----------------------------------------------------

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)

pd.DataFrame(mlp.loss_curve_).plot()
plt.xlabel("epochs")
plt.ylabel("loss_")
plt.show()

#%%#------------------------SVM Modeling-----------------------------------------------------

from sklearn.svm import SVC
import seaborn as sns

# creating the classifier object
clf = SVC(kernel="linear")
# performing training
clf.fit(X_train_flatten, y_train)
# predicton on test
y_pred = clf.predict(X_test_flatten)
print("\n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
print("\n")

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = ['no_t', 'pituitary_t', 'meningioma_t', 'glioma_t']

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False,annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()

#%%#------------------------KNN Modeling-----------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

clf = KNeighborsClassifier(n_neighbors=4)
# performing training
clf.fit(X_train_flatten, y_train)
# predicton on test
y_pred = clf.predict(X_test_flatten)
print("\n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

import seaborn as sns
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = ['no_t', 'pituitary_t', 'meningioma_t', 'glioma_t']

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False,annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()


#%%#------------------------Naive Bayes Modeling-----------------------------------------------------

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

clf.fit(X_train_flatten, y_train)
# predicton on test
y_pred = clf.predict(X_test_flatten)
print("\n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = ['no_t', 'pituitary_t', 'meningioma_t', 'glioma_t']

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False,annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()