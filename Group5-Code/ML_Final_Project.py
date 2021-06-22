#%%#-----------------------import packages--------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop,SGD,Adagrad,Adadelta,Adamax,Adam
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,MaxPooling2D
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle

from numpy.random import seed
seed(1)

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
counts = []
for file in os.listdir(directory):
    path = os.path.join(directory,file)
    count=0
    for im in os.listdir(path):
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(128,128))
        image=img_to_array(image)
        image=image/255.0
        dataset.append(image)
        labels.append(mapping[file])
        count += 1
    counts.append(count)
print(len(dataset), len(labels))
print(dict(zip(File,counts)))

#%%#---------------------------define X, y for SVM, KNN, MLP-----------------------------------------------------

X = np.array(dataset)
y = np.array(labels)
print(X.shape)
print(y.shape)

#%%#---------------------------define X, y for ANN, CNN------------------------------------------------------

labels1 = to_categorical(labels)  # array to matrix
X = np.array(dataset)
y = np.array(labels1)
print(X.shape)
print(y.shape)

#%%#------------------------data shuffling----------------------------------------------------------

X, y = shuffle(X, y, random_state=100)

#%%#------------------------Data Augmentation ----------------------------------------------------------
# Image Data Augmentation: Image data augmentation is a technique that can be used to artificially expand the size of a training dataset by creating modified versions of images in the dataset. It uses techniques such as flipping, zooming, padding, cropping, etc.
# Data augmentation makes the model more robust to slight variations, and hence prevents the model from overfitting.
X_datagenerator = ImageDataGenerator(rotation_range=20,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True)
X_datagenerator.fit(X)
print(X.shape)
print(y.shape)

#%%#------------------------data split----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)

#%%#------------------------reshape data from 3D to 2D----------------------------------------------------------------

X_train_flatten = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
X_test_flatten = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3])
print("X_train_flatten",X_train_flatten.shape)
print("X_test_flatten",X_test_flatten.shape)

#%%#------------------------ANN----------------------------------------------------------

model = Sequential()
model.add(Dense(500, activation='sigmoid', input_shape=(49152,)))
model.add(Dropout(0.2))
# #
model.add(Dense(500, activation='sigmoid'))
model.add(Dropout(0.2))
#
model.add(Dense(20, activation='sigmoid'))
model.add(Dropout(0.2))
# # #
# model.add(Dense(20, activation='sigmoid'))
# model.add(Dropout(0.2))
# #
# model.add(Dense(20, activation='sigmoid'))
# model.add(Dropout(0.2))
# # # #
# model.add(Dense(20, activation='sigmoid'))
# model.add(Dropout(0.2))

model.add(Dense(4, activation='softmax'))

#model.summary()
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0002),metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.01),metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',optimizer=Adagrad(lr=0.01),metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',optimizer=Adadelta(lr=1),metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',optimizer=Adamax(lr=0.002),metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=20)
history = model.fit(X_train_flatten, y_train, batch_size=64, epochs=500, verbose=1, validation_split=0.10,
                    #callbacks=[es]
                    )
score = model.evaluate(X_test_flatten, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#%%#------------------------ANN model prediction----------------------------------------------------------

pred = model.predict(X_test_flatten)
y_pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)

Class_Report = classification_report(y_test_new,y_pred)
print("Classification Report:")
print(Class_Report)

conf_matrix = confusion_matrix(y_test_new, y_pred)
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

#%%#------------------------CNN model building and training----------------------------------------------------------

Model = Sequential()
Model.add(Conv2D(filters = 64, kernel_size = (3,3),
                 activation ='relu', input_shape = (128,128,3)))
Model.add(MaxPool2D(pool_size=(2,2)))
Model.add(Dropout(0.2))

Model.add(Conv2D(filters = 128, kernel_size = (3,3),
                 activation ='relu'))
Model.add(MaxPool2D(pool_size=(2,2)))
Model.add(Dropout(0.2))

Model.add(Conv2D(filters = 128, kernel_size = (3,3),
                 activation ='relu'))
Model.add(MaxPool2D(pool_size=(2,2)))
Model.add(Dropout(0.2))

Model.add(Conv2D(filters = 128, kernel_size = (3,3),
                 activation ='relu'))
Model.add(MaxPool2D(pool_size=(2,2)))
Model.add(Dropout(0.2))

Model.add(Flatten())
Model.add(Dense(128, activation = "relu"))
Model.add(Dropout(0.25))

Model.add(Dense(4, activation = "softmax"))

Model.compile(optimizer=RMSprop(lr=0.001),
             loss="categorical_crossentropy",
             metrics=["accuracy"])

Call_Back = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=2,mode="min")

CNN_Model = Model.fit(X_train,y_train,epochs=50,batch_size=32,callbacks=Call_Back)

#%%#------------------------CNN model evaluation----------------------------------------------------------

Dict_Summary = pd.DataFrame(CNN_Model.history)
Dict_Summary.plot()
plt.title("LOSS & ACCURACY")
plt.xlabel("EPOCH")
plt.ylabel("Loss & Acc")
plt.show()

#%%#------------------------CNN model prediction----------------------------------------------------------

y_pred = Model.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)

import seaborn as sns
conf_matrix = confusion_matrix(y_test_new, y_pred)
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

#%%#------------------------classifier--------------------------------------------------------

mlp = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=500, activation='relu',early_stopping=False,momentum=0.9,solver='adam')
#mlp = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=500, activation='relu',early_stopping=False,momentum=0.9,solver='adam' )
#mlp = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=500, activation='relu',early_stopping=False,momentum=0.9,solver='adam')
#mlp = MLPClassifier(hidden_layer_sizes=(200,200),max_iter=500, activation='relu',early_stopping=False,momentum=0.9,solver='adam' )
#mlp = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=5000, activation='relu',early_stopping=False,momentum=0.9,solver='adam' )
#mlp = MLPClassifier(hidden_layer_sizes=(20,20,20,20),max_iter=500, activation='relu',early_stopping=False,momentum=0.9,solver='adam' )
#mlp = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=500, activation='logistic',early_stopping=False,momentum=0.9,solver='adam' )
#mlp = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=500, activation='relu',early_stopping=True,momentum=0.9,solver='adam' )
#mlp = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=500, activation='relu',early_stopping=False,momentum=0.9,solver='sgd' )
#mlp = MLPClassifier(hidden_layer_sizes=(200,200,20,20,20,20),max_iter=500, activation='relu',early_stopping=False,momentum=0.9,solver='adam' )
#mlp = MLPClassifier(hidden_layer_sizes=(200, 20, 20, 20, 20, 20,20,20,10),max_iter=500, activation='relu',early_stopping=False,momentum=0.9,solver='adam' )

#%%#------------------------training and prediction----------------------------------------

mlp.fit(X_train_flatten, y_train)
y_pred = mlp.predict(X_test_flatten)

#%#------------------------evaluation and loss-----------------------------------------------------

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)

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

pd.DataFrame(mlp.loss_curve_).plot()
plt.xlabel("epochs")
plt.ylabel("loss_")
plt.show()

#%%#------------------------SVM Modeling-----------------------------------------------------

# creating the classifier object
clf = SVC(kernel="poly")
# performing training
clf.fit(X_train, y_train)
# predicton on test
y_pred = clf.predict(X_test)
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

clf = KNeighborsClassifier(n_neighbors=4)
# performing training
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


#%%#------------------------Naive Bayes Modeling-----------------------------------------------------

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