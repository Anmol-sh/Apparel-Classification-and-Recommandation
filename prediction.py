import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import cv2
import config

def predict(image_path):
    # load the class_indices saved in the earlier step
    class_dictionary = np.load(config.classIndex).item()
    num_classes = len(class_dictionary)

    # add the path to your test image below
    # image_path = 'data/validation/women_tops/1424.jpg'

    orig = cv2.imread(image_path)
    print("classification of ",image_path)
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255
    image = np.expand_dims(image, axis=0)

    # build the VGG19 network
    model = applications.VGG19(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG19 model
    bottleneck_prediction = model.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.load_weights(config.top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final classification
    class_predicted = model.predict_classes(bottleneck_prediction)
    probabilities = model.predict_proba(bottleneck_prediction)
    inID = class_predicted[0]
    inv_map = {v: k for k, v in class_dictionary.items()}
    label = inv_map[inID]

    # get the prediction label
    print("Class/Category: {}".format(label))
    return label
