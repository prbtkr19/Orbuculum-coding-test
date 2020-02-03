#importing libraries needed during implementing project
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm

#loading training data
train = pd.read_csv('F:\\task\\training_set\\annotation.csv')

# creating validations set from training set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
 
#building the covolutional neural network
#importing keras library
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialising CNN
classifier=Sequential()
# step 1 Convolution
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),activation='relu'))
# i have used relu to make it non linear

#step2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step3 flatten
classifier.add(Flatten())

#step 4 Full connection
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='softmax'))
#since there are many than two binary classifier so i have used softmax

#compiling the CNN
#used categorical loss because there are more than two outputs
classifier.compile(optimizer='adam',loss = 'categorical_crossentropy'
 ,metrics=['accuracy'])

#fitting CNN to images
#importing class generator library
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory("F:\task\annotation.csv"
                                                    target_size=(64,64),
                                                    batch_size=32,
                                                    class_mode='categorical')

test_set = test_datagen.flow_from_directory('F:\task\test_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=800)


# making new predictions
import numpy as np
from keras.preprocessing import image
test_image=image.load_img()
test_image=image.img_to_array(test_image)
test_image=np.expand_dim(test_image,axis=0)
result=classifier.predict(test_image,axis=0)
training_set.class_indices


# calculating F1 score
from sklearn.metrics import f1_score
score=f1_score(y_true, y_pred, average=‘samples’)






