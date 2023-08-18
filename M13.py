from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
from os import listdir


#stuff='CardBoard Comparison'
#stuff='Wood Comparison'
stuff='Metal Comparison'

model='M13'
path='D:/U-NET_Project/UNet Paper Comparison/UNet Paper Comparison/\
{}/TrainDateLabels4Dnn.csv'.format(stuff)
path1='D:/U-NET_Project/UNet Paper Comparison/UNet Paper Comparison/\
{}/Traindnn/'.format(stuff)
path2='D:/U-NET_Project/UNet Paper Comparison/UNet Paper Comparison/\
{}/Test&Validation/'.format(stuff)
path3='D:/U-NET_Project/UNet Paper Comparison/UNet Paper Comparison/\
{}/Test&ValidationDataLabels.csv'.format(stuff)
path4='D:/U-NET_Project/UNet Paper Comparison/UNet Paper Comparison/\
{}/{}.csv'.format(stuff,model)
path5='D:/U-NET_Project/UNet Paper Comparison/UNet Paper Comparison/\
{}/Validation.csv'.format(stuff)

#loading images for CNN
df_train = pd.read_csv(path)
train_image = []
for i in df_train['FileName'] :
    img = image.load_img(path1 + str(i),target_size=(256, 256),color_mode='grayscale')
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)       
X_train = np.array(train_image)
# Creating the response variable
y_train=df_train['Label'].values
# y_train = to_categorical(y_train)

print(f"Number of records: {df_train.shape[0]:,}")
print(f"shape of inputs: {X_train.shape}")
print(f"lenth of y:{len(y_train)}")

#loading image for CNN
df_test = pd.read_csv(path3)
test_image = []
for i in df_test['FileName'] :
    img = image.load_img(path2 + str(i),target_size=(256, 256),color_mode='grayscale')
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)       
X_test = np.array(test_image)
y_test=df_test['Label'].values
# y_test = to_categorical(y_test)
print(f"Number of records: {df_test.shape[0]:,}")
print(f"Number of records: {X_test.shape}")
print(f"lenth of y:{len(y_test)}")


#loading image for CNN
df_val = pd.read_csv(path5)
test_image = []
for i in df_val['FileName'] :
    img = image.load_img(path2 + str(i),target_size=(256, 256),color_mode='grayscale')
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)       
X_v = np.array(test_image)
y_v=df_val['Label'].values
# y_test = to_categorical(y_test)
print(f"Number of records: {df_val.shape[0]:,}")
print(f"Number of records: {X_v.shape}")
print(f"lenth of y:{len(y_v)}")


# DNN 
import numpy as np
from os import listdir
import tensorflow as tf
import pickle
from tensorflow.keras.layers import Dense, Flatten, concatenate,Input,Conv2D
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.core import Activation, Flatten, Dropout, Dense, Reshape
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing import image
#change this line code 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

def larger_model():
    # create model
    model = Sequential()
    model.add(Conv2D(20, (5, 5), input_shape=(256,256,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(10, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
    
    # Compile model
    return model
model=larger_model()
checkpointer = ModelCheckpoint('model-cardboard_M13.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, y_train, validation_data=(X_v, y_v), batch_size=200, epochs=10, 
                    callbacks=[checkpointer])

# Load the previously saved weights
model.load_weights('model-cardboard_M13.h5')
import sklearn.metrics as metrics
y_pred = model.predict(X_test)  
dataset = pd.DataFrame({'M13': y_pred.reshape(-1)})
dataset.to_csv(path4)                   