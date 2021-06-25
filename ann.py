# written in visual studio code

# necessary libraries
from keras.activations import linear
from keras.metrics import accuracy
import pandas as pd
import numpy as np

#  reading csv file by help of pandas library
train = pd.read_csv("/home/vozcelik/Downloads/mnist_train.csv")
test = pd.read_csv("/home/vozcelik/Downloads/mnist_test.csv")

# reshaping the data
train.values.reshape(-1,1)
test.values.reshape(-1,1)

print(train.shape)
print(test.shape)

# splitting our data to train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train,test, test_size=0.33, random_state=15)

from keras.models import Sequential
from keras.layers import Dense

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> c r e a t i n g   o u r   m o d e l (   s i g mo i d   a c t i v a t i o n   )  <<<<<<<<<<<<<<<<<<<<<<<<
model_sigmoid = Sequential()

# as in the question required, hidden layer's activation func. is sigmoid and output layer same as hidden layer
model_sigmoid.add(Dense(8, activation='sigmoid'))
model_sigmoid.add(Dense(1, activation='sigmoid'))

# widely adam optimizer is used 
model_sigmoid.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# epochs size just 5 to finish shortly ( you can increase it to get higger accuracy )
model_sigmoid.fit(X_train, y_train, batch_size=8, epochs=5)

# convergence cerve by help of sklearn ( it can be done by scipy curve_fit as well but i dont know how actually use it )
from sklearn.metrics import plot_roc_curve
svc_disp = plot_roc_curve(model_sigmoid, X_test, y_test)


test_loss, test_acc = model_sigmoid.evaluate(X_test, y_test)
print("accuracy is : ", test_acc," and loss is : ",test_loss)

# to calculate accuracy and f1 score
# here is the labrary for it
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(X_test, y_test)

from sklearn.metrics import f1_score
f1_scr = f1_score(X_test, y_test)

print("here is the result\n accuracy is:",accuracy, "and f1 score is: ",f1_scr)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   r e l u   a c t i v a t i o n   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

model_ReLU = Sequential()

model_ReLU.add(Dense(8, activation='rule'))
model_ReLU.add(Dense(1, activation='rule'))

model_ReLU.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model_ReLU.fit(X_train, y_train, batch_size=8, epochs=5)

# convergence cerve by help of sklearn 

from sklearn.metrics import plot_roc_curve
svc_disp = plot_roc_curve(model_ReLU, X_test, y_test)


test_loss, test_acc = model_ReLU.evaluate(X_test, y_test)
print("this is for relu activation function")
print("accuracy: ", test_acc," and score : ",test_loss)

# to calculate accuracy and f1 score
# here is the labrary for it
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(X_test, y_test)

from sklearn.metrics import f1_score
f1_scr = f1_score(X_test, y_test)

print("here is the result\n accuracy is:",accuracy, "and f1 score is: ",f1_scr)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>> a c t i v a t i o n   f u n c t i o n    f o r    t a n h    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

model_tanh = Sequential()

model_tanh.add(Dense(8, activation='tanh'))
model_tanh.add(Dense(1, activation='tanh'))

model_tanh.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# epochs size just 5 to make finish ( you can increase it to get higger accuracy )
model_tanh.fit(X_train, y_train, batch_size=8, epochs=5)

# convergence cerve by help of sklearn 

from sklearn.metrics import plot_roc_curve
svc_disp = plot_roc_curve(model_ReLU, X_test, y_test)


test_loss, test_acc = model_tanh.evaluate(X_test, y_test)
print("this is for tanh activation function")
print("accuracy: ", test_acc," and score : ",test_loss)

# to calculate accuracy and f1 score
# here is the labrary for it
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(X_test, y_test)

from sklearn.metrics import f1_score
f1_scr = f1_score(X_test, y_test)

print("here is the result\n accuracy is:",accuracy, "and f1 score is: ",f1_scr)