'''
Using Bottleneck Features for Multi-Class Classification in Keras
We use this technique to build powerful (high accuracy without overfitting) Image Classification systems with small
amount of training data.
The code was tested on Python 3.5, with the following library versions,
Keras 2.0.6
TensorFlow 1.2.1
OpenCV 3.2.0
This should work with Theano as well, but untested.
'''
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import cv2
import config
from shutil import copyfile, rmtree, copytree     #################
from os import remove, path                       #################

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = config.top_model_weights_path
train_data_dir = config.trainFile
validation_data_dir = config.validationFile

# number of epochs to train top model
epochs = config.epochs
# batch size used by flow_from_directory and predict_generator
batch_size = config.batchSize


def save_bottlebeck_features():
    print("In save_bottlebeck_features")
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    # print("generator.filnames length: ",len(generator.filenames))
    # print("generator.class_indices: ",generator.class_indices)
    # print("generator.class_indices length: ",len(generator.class_indices))

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)

    np.save(config.trainFeature, bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)

    np.save(config.validationFeature,
            bottleneck_features_validation)

def train_top_model():
    print("In train_top_model")
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # save the class indices to use use later in predictions
    np.save(config.classIndex, generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load(config.trainFeature)

    # get the class lebels for the training data, in the original order
    train_labels = generator_top.classes

    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load(config.validationFeature)

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    ans = input("would you like to save the bottleneck features?(y/n)")              ###############
    if ans=='y':
        print("saving files")
        remove('backup/'+config.top_model_weights_path)
        copyfile(config.top_model_weights_path, 'backup/'+config.top_model_weights_path)

        remove('backup/'+config.trainFeature)
        copyfile(config.trainFeature, 'backup/'+config.trainFeature)

        remove('backup/'+config.validationFeature)
        copyfile(config.validationFeature, 'backup/'+config.validationFeature)

        remove('backup/'+config.classIndex)
        copyfile(config.classIndex, 'backup/'+config.classIndex)

        rmtree('backup/' + config.trainFile)
        copytree(config.trainFile, 'backup/' + config.trainFile)

        rmtree('backup/' + config.validationFile)
        copytree(config.validationFile, 'backup/' + config.validationFile)

        plot = input("Filename:- ")
        if path.exists(plot):
            remove(plot)
        plt.savefig('foo.png')

def train_model():
    save_bottlebeck_features()
    train_top_model()

if __name__=='__main__':
    train_model()