
# coding: utf-8

# In[1]:


#import all the necessary libraries

import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import cv2


# In[2]:


#import the data from MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Reshape it
x_train = x_train.reshape(60000,784)
x_test  = x_test.reshape(10000,784)
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')


y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)


# In[3]:


#Create a model

model = Sequential([
    Dense(32, input_shape=(784,), activation = 'tanh'),
    Dense(10, activation = "softmax")
])


# In[4]:


#Compile the model

model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])


# In[5]:


#Here we train it

model.fit(x_train,y_train, batch_size=10, epochs=10, shuffle=True, verbose=2,validation_data=(x_test, y_test))


#To see how much accuracy we have and how much of the loss

score = model.evaluate(x_train,y_train)
print('Test loss = ',score[0])
print('Test Accuracy = ', score[1])


# In[6]:


#Import the image from the link: https://goo.gl/rkRn4k

img_brg = cv2.imread('image.jpg',cv2.IMREAD_COLOR)

img_rbg = cv2.cvtColor(img_brg, cv2.COLOR_BGR2RGB)

plt.imshow(img_rbg), plt.axis("off")
plt.show()


# In[7]:


# Reshape it and give it to our model

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

img_resized = img.resize(28,28)

img_reshaped = img.reshape(1,784)


# In[8]:


#Predict the digit from image

prediction = model.predict(img_reshaped)

print('Prediction Score:\n',prediction[0])
thresholded = (prediction>0.5)*1
print('\nThresholded Score:\n',thresholded[0])
print('\nPredicted Digit:\n',np.where(thresholded == 1)[1][0])

