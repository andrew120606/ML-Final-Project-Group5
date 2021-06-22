#code from :https://machinelearningmastery.com

# load all images in a directory
from os import listdir
from matplotlib import image
import os
import glob
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from numpy import asarray

train_dir = '/Volumes/GoogleDrive/My Drive/ML6202-summer2021/BrainCancerData/brain-tumor-classification-dataset/Training/'
lbls={'no_tumor_subset':0, 'pituitary_tumor_subset':1, 'meningioma_tumor_subset':2, 'glioma_tumor_subset':3}

# load all images in a directory
train_images = list()
train_labels = np.array([])

# read in the data
for tumorlbs in lbls:
    for file in glob.glob(train_dir + tumorlbs+'/*.jpg'):
        # load image
        #btumortrain_data_ml = image.imread(file)
        # store loaded image
        #train_images.append(btumortrain_data)
        #train_labels.append(lbls[tumorlbs])
        #print('> loaded %s %s %s %s' % (file, btumortrain_data_ml.dtype, btumortrain_data_ml.shape,lbls[tumorlbs]))
        btumortrain_img = load_img(file, color_mode='rgb', target_size=(150,150))

        print(type(btumortrain_img))


        # convert to numpy array
        btumortrain_data = img_to_array(btumortrain_img)

        #btumortrain_data_test = img_to_array(btumortrain_img_test)
        #print(btumortrain_data)
        #print(btumortrain_data.dtype)
        #print(btumortrain_data.shape)

        #print(btumortrain_data_test.dtype)
        #print(btumortrain_data_test.shape)
        train_images.append(btumortrain_data.flatten())
        train_labels=np.append(train_labels,lbls[tumorlbs])

#preprocess
# define one hot encoding
encoder = OneHotEncoder(sparse=False)
# transform data
onehot = encoder.fit_transform(train_labels.reshape(-1, 1))#.reshape(-1, 1)
btumortrain_data_in , btumortrain_data_lbl_in= shuffle(train_images,onehot, random_state=100)
btumortrain_data_tr,btumortrain_data_vl,btumortrain_data_lbl_tr,btumortrain_data_lbl_vl=train_test_split(btumortrain_data_in,btumortrain_data_lbl_in,test_size=0.3, stratify=btumortrain_data_lbl_in, random_state=100)

sc = StandardScaler()
btumortrain_data_tr_std = sc.fit_transform(btumortrain_data_tr)
btumortrain_data_vl_std = sc.transform(btumortrain_data_vl)

mlp = MLPClassifier(hidden_layer_sizes=(13,13,4),max_iter=5000, activation='logistic',early_stopping=True,momentum=0.95,solver='sgd' )
#mlp.fit(X_train,y_train)

#predictions = mlp.predict(X_test)

#print(confusion_matrix(y_test,predictions))

# from PIL import Image
# from numpy import asarray
# # load the image
# image = Image.open('/Volumes/GoogleDrive/My Drive/ML6202-summer2021/BrainCancerData/brain-tumor-classification-dataset/Training/no_tumor/1.jpg')
# # convert image to numpy array
# data = asarray(image)
# # summarize shape
# print(data.shape)
# # create Pillow image
# image2 = Image.fromarray(data)
# # summarize image details
# print(image2.format)
# print(image2.mode)
# print(image2.size)




# #code borrowed from
#
# import tensorflow as tf
# import os
# import glob
#
# #https://towardsdatascience.com/image-recognition-with-machine-learning-on-python-image-processing-3abe6b158e9a
#
# def decode_image(filename, image_type, resize_shape, channels):
#     value = tf.io.read_file(filename)
#     if image_type == 'png':
#         decoded_image = tf.image.decode_png(value, channels=channels)
#     elif image_type == 'jpeg':
#         decoded_image = tf.image.decode_jpeg(value, channels=channels)
#     else:
#         decoded_image = tf.image.decode_image(value, channels=channels)
#
#     if resize_shape is not None and image_type in ['png', 'jpeg']:
#         decoded_image = tf.image.resize(decoded_image, resize_shape)
#
#     return decoded_image
#
#
# def get_dataset(image_paths, image_type, resize_shape, channels):
#     filename_tensor = tf.constant(image_paths)
#     dataset = tf.data.Dataset.from_tensor_slices(filename_tensor)
#
#     def _map_fn(filename):
#         decode_images = decode_image(filename, image_type, resize_shape, channels=channels)
#         return decode_images
#
#     map_dataset = dataset.map(_map_fn)  # we use the map method: allow to apply the function _map_fn to all the
#     # elements of dataset
#     return map_dataset
#
#
# def get_image_data(image_paths, image_type, resize_shape, channels):
#     dataset = get_dataset(image_paths, image_type, resize_shape, channels)
#     iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
#     next_image = iterator.get_next()
#
#     return next_image
#
# train_dir = '/Volumes/GoogleDrive/My Drive/ML6202-summer2021/BrainCancerData/brain-tumor-classification-dataset/Training/'
# notumortrainfl=[]
# gliomanotumortrainfl=[]
# pituitarytrainfl=[]
# meningiomatrainfl=[]
#
# for file in glob.glob(train_dir + 'no_tumor_subset/*.jpg'):
#     notumortrainfl+=[file]
# print(notumortrainfl)
#
# for file in glob.glob(train_dir + 'glioma_tumor_subset/*.jpg'):
#     gliomanotumortrainfl+=[file]
# print(gliomanotumortrainfl)
#
# for file in glob.glob(train_dir + 'meningioma_tumor_subset/*.jpg'):
#     meningiomatrainfl+=[file]
# print(meningiomatrainfl)
#
# for file in glob.glob(train_dir + 'pituitary_tumor_subset/*.jpg'):
#     pituitarytrainfl+=[file]
# print(pituitarytrainfl)
#
# notumortraindata=get_image_data(notumortrainfl,'jpg',0,0)
# gliomanotumortraindata=get_image_data(gliomanotumortrainfl,'jpg',0,0)
# pituitarytraindata=get_image_data(pituitarytrainfl,'jpg',0,0)
# meningiomatraindata=get_image_data(meningiomatrainfl,'jpg',0,0)
