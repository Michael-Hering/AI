from __future__ import print_function
import keras
import utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import os

import numpy as np

from PIL import Image

import re
from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt

def getVGGFeatures(img, layerName, base_model): # Pass in the model, so we don't have to get it every time

  # base_model = VGG16(weights='imagenet')
  model = Model(inputs=base_model.input, outputs=base_model.get_layer(layerName).output)

  img = img.resize((224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  internalFeatures = model.predict(x)

  return internalFeatures

def cropImage(image, x1, y1, x2, y2):
  croppedImage = image.crop((x1, y1, x2, y2))
  return croppedImage

def standardizeImage(image, x, y):
  standardizedImage = image.resize(tuple([x, y]))
  return standardizedImage

def preProcessImages():
  if not os.path.exists("cropped"):
    os.makedirs("cropped")
  if not os.path.exists("standardized"):
    os.makedirs("standardized")

  for line in open("image_metadata.txt", "r"):
    items = line.split(',')
    dimensions = items[1:]
    dimensions = [int(d) for d in dimensions]
    filename = items[0]
    path_to_file = "uncropped/" + filename
    try:
      raw_image = Image.open(path_to_file)
      print("processing: ", filename, dimensions)
      
    except:
      print("could not open image:", filename, dimensions)
      continue

    cropped_image = cropImage(raw_image, dimensions[0], dimensions[1], dimensions[2], dimensions[3])
    standardized_image = standardizeImage(cropped_image, 60, 60)
    cropped_image.save("cropped/" + filename)
    standardized_image.save("standardized/" + filename)

def visualizeWeight():
  #You can change these parameters if you need to
  utils.raiseNotDefined()

def trainFaceClassifier(preProcessedImages, labels):
  # building a fully connected network with a single hidden layer
  model = Sequential()
  model.add(Dense(512, input_shape=(preProcessedImages.shape[1],), activation='relu'))
  model.add(Dense(6, activation='softmax'))                      
  model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

  # training the model and saving metrics in history
  history = model.fit(preProcessedImages, labels, validation_split = 0.33,
    batch_size=128, epochs=100,
    verbose=2)

  return model, history


def trainFaceClassifier_VGG(extractedFeatures, labels):
  # building a fully connected network with a single hidden layer
  model = Sequential()
  model.add(Dense(256, input_shape=(extractedFeatures.shape[1],), activation='relu'))                      
  model.add(Dense(6, activation='softmax'))                      
  model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

  # training the model and saving metrics in history
  history = model.fit(extractedFeatures, labels, validation_split=0.33,
    batch_size=16, epochs=20,
    verbose=2)

  return model, history


if __name__ == '__main__':
  # Preprocess images and store proccessed images for later use
  # preProcessImages()

  # input image dimensions
  img_rows, img_cols = 60, 60

  # Map labels to integers
  acts= ['butler', 'radcliffe', 'vartan', 'bracco', 'gilpin', 'harmon']
  acts_labels = {}
  for k in range(len(acts)):
    acts_labels[acts[k]] = k




###############################
### Deep Learning framework ###
###############################


  # X_full = []
  # y_full = []

  # for file_name in os.listdir("standardized"):
  #   print("Gathering features for: ", file_name)
  #   standardized_image = Image.open("standardized/"+file_name)
  #   label = ""
  #   for a in acts: # Get the label from the filename with simple regex
  #     if re.search(a, file_name) != None:
  #       label = acts_labels[a]
    
  #   # Store features
  #   bw_image = standardized_image.convert('L')
  #   X_full.append(np.array(bw_image))
  #   y_full.append(label)

  # # Convert feature lists to numpy arrays
  # X_full = np.array(X_full)
  # y_full = np.array(y_full)

  # # split the data between train and test sets
  # X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.33, stratify=y_full)

  # # build the input vector from the 60x60 pixels
  # X_train = X_train.reshape(X_train.shape[0], img_rows * img_cols)
  # X_test = X_test.reshape(X_test.shape[0], img_rows * img_cols)
  # X_train = X_train.astype('float32')
  # X_test = X_test.astype('float32')

  # # normalize the data to help with the training
  # X_train /= 255
  # X_test /= 255

  # # one-hot encoding using keras' numpy-related utilities
  # y_train = keras.utils.to_categorical(y_train, num_classes=6)
  # y_test = keras.utils.to_categorical(y_test, num_classes=6)


  # # Train the model with the training data
  # model, history = trainFaceClassifier(X_train, y_train)

  # score = model.evaluate(X_test, y_test, verbose=0)
  # print('Test loss:', score[0])
  # print('Test accuracy:', score[1])

  # fig, ax = plt.subplots(1,1)
  # print(history.history.keys())
  # ax.plot(history.history['loss'], label="loss")
  # ax.plot(history.history['val_loss'], label="validation loss")
  # ax.set_title("Loss vs Epoch")
  # ax.set_xlabel("Epoch")
  # ax.set_ylabel("Loss")
  # ax.legend()
  # plt.show()


#####################################
### Transfer Learning using VGG16 ###
#####################################

  # base_model = VGG16(weights='imagenet')

  # X_vgg_full = []
  # y_vgg_full = []

  # for file_name in os.listdir("standardized"):
  #   print("Gathering features for: ", file_name)
  #   standardized_image = Image.open("standardized/"+file_name)

  #   # Store features
  #   if standardized_image.mode == "RGB": # Only use color images
  #     label = ""
  #     for a in acts: # Get the label from the filename with simple regex
  #       if re.search(a, file_name) != None:
  #         label = acts_labels[a]

  #     features = getVGGFeatures(standardized_image, "block5_pool", base_model)
  #     X_vgg_full.append(features)
  #     y_vgg_full.append(label)

  

  # # Convert feature lists to numpy arrays
  # X_vgg_full = np.array(X_vgg_full)
  # y_vgg_full = np.array(y_vgg_full)

  # # split the data between train and test sets
  # X_vgg_train, X_vgg_test, y_vgg_train, y_vgg_test = train_test_split(X_vgg_full, y_vgg_full, test_size=0.33, stratify=y_vgg_full)

  # # build the input vector from the features
  # print(X_vgg_train.shape)
  # X_vgg_train = X_vgg_train.reshape(X_vgg_train.shape[0], X_vgg_train.shape[2] * X_vgg_train.shape[3] * X_vgg_train.shape[4])
  # X_vgg_test = X_vgg_test.reshape(X_vgg_test.shape[0], X_vgg_test.shape[2] * X_vgg_test.shape[3] * X_vgg_test.shape[4])
  # X_vgg_train = X_vgg_train.astype('float32')
  # X_vgg_test = X_vgg_test.astype('float32')

  # # normalize the data to help with the training
  # # X_train /= 255
  # # X_test /= 255

  # # one-hot encoding using keras' numpy-related utilities
  # y_vgg_train = keras.utils.to_categorical(y_vgg_train, num_classes=6)
  # y_vgg_test = keras.utils.to_categorical(y_vgg_test, num_classes=6)

  # # Train the model with the training data
  # transfer_model, transfer_model_history = trainFaceClassifier_VGG(X_vgg_train, y_vgg_train)

  # transfer_score = transfer_model.evaluate(X_vgg_test, y_vgg_test, verbose=0)
  # print('Test loss:', transfer_score[0])
  # print('Test accuracy:', transfer_score[1])

  # fig, ax = plt.subplots(1,1)
  # print(transfer_model_history.history.keys())
  # ax.plot(transfer_model_history.history['loss'], label="loss")
  # ax.plot(transfer_model_history.history['val_loss'], label="validation loss")
  # ax.set_title("Loss vs Epoch")
  # ax.set_xlabel("Epoch")
  # ax.set_ylabel("Loss")
  # ax.legend()
  # plt.show()


#######################
### Expirimentation ###
#######################

  # base_model = VGG16(weights='imagenet')

  # X_vgg_full = []
  # y_vgg_full = []

  # for file_name in os.listdir("standardized"):
  #   print("Gathering features for: ", file_name)
  #   standardized_image = Image.open("standardized/"+file_name)

  #   # Store features
  #   if standardized_image.mode == "RGB": # Only use color images
  #     label = ""
  #     for a in acts: # Get the label from the filename with simple regex
  #       if re.search(a, file_name) != None:
  #         label = acts_labels[a]

  #     features = getVGGFeatures(standardized_image, "block5_pool", base_model)
  #     X_vgg_full.append(features)
  #     y_vgg_full.append(label)

  

  # # Convert feature lists to numpy arrays
  # X_vgg_full = np.array(X_vgg_full)
  # y_vgg_full = np.array(y_vgg_full)

  # # split the data between train and test sets
  # X_vgg_train, X_vgg_test, y_vgg_train, y_vgg_test = train_test_split(X_vgg_full, y_vgg_full, test_size=0.33, stratify=y_vgg_full)

  # # build the input vector from the features
  # print(X_vgg_train.shape)
  # X_vgg_train = X_vgg_train.reshape(X_vgg_train.shape[0], X_vgg_train.shape[2] * X_vgg_train.shape[3] * X_vgg_train.shape[4])
  # X_vgg_test = X_vgg_test.reshape(X_vgg_test.shape[0], X_vgg_test.shape[2] * X_vgg_test.shape[3] * X_vgg_test.shape[4])
  # X_vgg_train = X_vgg_train.astype('float32')
  # X_vgg_test = X_vgg_test.astype('float32')

  # # normalize the data to help with the training
  # # X_train /= 255
  # # X_test /= 255

  # # one-hot encoding using keras' numpy-related utilities
  # y_vgg_train = keras.utils.to_categorical(y_vgg_train, num_classes=6)
  # y_vgg_test = keras.utils.to_categorical(y_vgg_test, num_classes=6)

  # dropout_rates = [0, 0.5]
  # num_layers = [1, 2]
  # num_nodes = [256, 512, 1024]

  # accuracies_dict = {}
  # histories_dict = {}

  # for dropout_rate in dropout_rates:
  #   for n_layers in num_layers:
  #     for n_nodes in num_nodes:
  #       # Build a model
  #         model = Sequential()
  #         for _ in range(n_layers):
  #           model.add(Dense(n_nodes, input_shape=(X_vgg_train.shape[1],), activation='relu'))    
  #           if (dropout_rate):
  #             model.add(Dropout(dropout_rate))

  #         model.add(Dense(6, activation='softmax'))                      
  #         model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

  #       # train the model and save metrics in history
  #         history = model.fit(X_vgg_train, y_vgg_train, validation_split=0.33,
  #         batch_size=128, epochs=20,
  #         verbose=2)

  #       # Get model accuracy
  #         transfer_score = model.evaluate(X_vgg_test, y_vgg_test, verbose=0)
  #         print('Test loss:', transfer_score[0])
  #         print('Test accuracy:', transfer_score[1])

  #       # Generate a key, <dropout rate>:<number of layers>:<number of nodes in each layer>
  #         key = str(dropout_rate) + ":" + str(n_layers) + ":" + str(n_nodes)

  #         accuracies_dict[key] = transfer_score[1]
  #         histories_dict[key] = history


  # # Plot our results
  # for key in histories_dict.keys():
  #   fig, ax = plt.subplots(1,1)
  #   history = histories_dict[key]
  #   ax.plot(history.history['loss'], label="loss")
  #   ax.plot(history.history['val_loss'], label="validation loss")
  #   ax.set_title("Loss vs Epoch \n" + key)
  #   ax.set_xlabel("Epoch")
  #   ax.set_ylabel("Loss")
  #   ax.legend()
  #   plt.show()

  # for key in accuracies_dict.keys():
  #   print(key + " accuracy: " + str(accuracies_dict[key]))