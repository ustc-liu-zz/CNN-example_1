# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import activations
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras import optimizers
from keras import initializers

# for training set
class_names_1 = ['-2', '-1', '0','1', '2']
# for test set
class_names_3 = ['-3','-2', '-1', '0','1', '2','3']

# loading data
temp = sio.loadmat('train_set.mat')
train_set = temp['Wh']
temp = sio.loadmat('train_label.mat')
train_label = temp['W']

temp = sio.loadmat('test_data_1.mat')
test_set_1 = temp['Wh']
temp = sio.loadmat('test_label_1.mat')
test_label_1 = temp['W']

temp = sio.loadmat('test_data_3.mat')
test_set_3 = temp['Wh']
temp = sio.loadmat('test_label_3.mat')
test_label_3 = temp['W']

# flat the label
train_label = train_label.flatten()
test_label_1 = test_label_1.flatten()
test_label_3 = test_label_3.flatten()

# reshape the train data
x = train_set.reshape(100000,2,33,1)
y = train_label.astype(int)

# shuffle
index = np.arange(100000)
np.random.shuffle(index)
X_train = x[index,:,:,:]
y_train = y[index]

# build model
model = Sequential()
# conv, no padding
model.add(Conv2D(40,(2,2),strides=(1,1),padding='valid',kernel_initializer=initializers.glorot_normal(seed=314),
	bias_initializer='zeros',activation="relu",input_shape=(2,33,1)))
#model.add(BatchNormalization())
model.add(Conv2D(1,(1,1),strides=1,padding='valid',kernel_initializer=initializers.glorot_normal(seed=314),
	bias_initializer='zeros',activation="relu",input_shape=(1,32)))
#model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(2,activation="relu",kernel_initializer=initializers.glorot_normal(seed=314),bias_initializer='zeros'))
model.add(Dense(1,activation="linear",kernel_initializer=initializers.glorot_normal(seed=314),bias_initializer='zeros'))

#adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#sgd = optimizers.SGD(lr=0.0005, momentum=0.9, decay=1e-6, nesterov=True)

model.compile(optimizer=tf.train.AdamOptimizer(),loss='mean_squared_error',metrics=['accuracy'])
#model.compile(optimizer=adam,loss='mean_squared_error',metrics=['accuracy'])
#model.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])

# print the model information
model.summary()
# training the model
history = model.fit(X_train, y_train,batch_size=50,epochs=200,validation_split=0.1,shuffle=True)

test_set_1 = test_set_1.reshape(10000,2,33,1)
test_set_3 = test_set_3.reshape(3000,2,33,1)

# evaluate test set 1
test_loss_1, test_acc_1 = model.evaluate(test_set_1, test_label_1)
print('Test accuracy for test set 1:', test_acc_1)

# evaluate test set 3
test_loss_3, test_acc_3 = model.evaluate(test_set_3, test_label_3)
print('Test accuracy for test set 3:', test_acc_3)

##########
predictions_1 = model.predict(test_set_1)
predictions_1 = predictions_1.flatten()

sio.savemat('predictions_1.mat',mdict = {'predictions_1':predictions_1})
#plt.hist(predictions_1)

predictions_1 = np.rint(predictions_1)
predictions_1 = predictions_1.astype(int)
test_label_1 = test_label_1.astype(int)

predictions_3 = model.predict(test_set_3)
predictions_3 = predictions_3.flatten()
sio.savemat('predictions_3.mat',mdict = {'predictions_3':predictions_3})


predictions_3 = np.rint(predictions_3)
predictions_3 = predictions_3.astype(int)
test_label_3 = test_label_3.astype(int)

# define plot functions
def plot_image_1(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i]+2, true_label[i]+2, img[i,:,:,0]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = predictions_array
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} ({})".format(class_names_1[predicted_label],
                                class_names_1[true_label]),
                                color=color)

def plot_value_array_1(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i]+2, true_label[i]+2
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(5), 1, color="#777777")
  plt.ylim([0, 1])
  predicted_label = predictions_array

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

def plot_image_3(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i]+3, true_label[i]+3, img[i,:,:,0]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = predictions_array
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} ({})".format(class_names_3[predicted_label],
                                class_names_3[true_label]),
                                color=color)

def plot_value_array_3(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i]+3, true_label[i]+3
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(7), 1, color="#777777")
  plt.ylim([0, 1])
  predicted_label = predictions_array

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')



# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 4
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image_3(i, predictions_3, test_label_3, test_set_3)
  #plt.subplot(num_rows, 2*num_cols, 2*i+2)
  #plot_value_array_3(i, predictions_3, test_label_3)

plt.figure()
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.figure()
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.show()
