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


model='M5'
path='D:/U-NET_Project/UNet Paper Comparison/UNet Paper Comparison/\
{}/TrainDataLabels.csv'.format(stuff)
path1='D:/U-NET_Project/UNet Paper Comparison/UNet Paper Comparison/\
{}/Train/'.format(stuff)
path2='D:/U-NET_Project/UNet Paper Comparison/UNet Paper Comparison/\
{}/Test&Validation/'.format(stuff)
path3='D:/U-NET_Project/UNet Paper Comparison/UNet Paper Comparison/\
{}/Test&ValidationDataLabels.csv'.format(stuff)
path4='D:/U-NET_Project/UNet Paper Comparison/UNet Paper Comparison/\
{}/{}.csv'.format(stuff,model)

df_train = pd.read_csv(path)
train_image = []
for i in df_train['FileName'] :
    img = image.load_img(path1 + str(i))
    img = image.img_to_array(img)
    img = img/255
    img = img.flatten()
    train_image.append(img)       
X_train = np.array(train_image)
# Creating the response variable
y_train=df_train['Label'].values
# y_train = to_categorical(y_train)

print(f"Number of records: {df_train.shape[0]:,}")
print(f"shape of inputs: {X_train.shape}")
print(f"lenth of y:{len(y_train)}")

df_test = pd.read_csv(path3)
test_image = []
for i in df_test['FileName'] :
    img = image.load_img(path2 + str(i))
    img = image.img_to_array(img)
    img = img/255
    img = img.flatten()
    test_image.append(img)       
X_test = np.array(test_image)
y_test=df_test['Label'].values
# y_test = to_categorical(y_test)
print(f"Number of records: {df_test.shape[0]:,}")
print(f"lenth of y:{len(y_test)}")

#Adaboost regression
from sklearn.ensemble import AdaBoostRegressor
adaboost_regressor = AdaBoostRegressor(random_state=0, n_estimators=100)
ada_model = adaboost_regressor.fit(X_train, y_train)
prediction_test_ada = ada_model.predict(X_test)
# Apply the model we created using the training data to the test data, and calculate the RSS.
print('RSS',((y_test - prediction_test_ada) **2).sum())
from sklearn import metrics
# Calculate the RMSE (Root Mean Squared Error)
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test,prediction_test_ada)))
print('score',ada_model.score(X_train, y_train))
dataset = pd.DataFrame({model: prediction_test_ada})
dataset.to_csv(path4)