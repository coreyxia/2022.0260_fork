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


model='M9'
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
    img = image.load_img(path1 + str(i),color_mode='grayscale')
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
    img = image.load_img(path2 + str(i),color_mode='grayscale')
    img = image.img_to_array(img)
    img = img/255
    img = img.flatten()
    test_image.append(img) 
    
X_test = np.array(test_image)

y_test=df_test['Label'].values
# y_test = to_categorical(y_test)
print(f"Number of records: {df_test.shape[0]:,}")
print(f"lenth of y:{len(y_test)}")

#linear regression model training
from sklearn import metrics
from sklearn import linear_model
lasso_alphas = np.linspace(0, 0.2, 21)
best_mse = 100000000000
best_alpha=0.5

for i in lasso_alphas:
    reg = linear_model.Lasso(alpha=i)
    reg.fit(X_train, y_train)
    test_predict=reg.predict(X_test)
    mse_train=metrics.mean_squared_error(y_test,test_predict)
    if best_mse > mse_train:
        best_mse = mse_train
        best_alpha = i
print(best_mse, best_alpha)


reg = linear_model.Lasso(alpha=best_alpha)
reg.fit(X_train, y_train)
test_predict=reg.predict(X_test)
dataset = pd.DataFrame({model: test_predict})
dataset.to_csv(path4)
print(f"Number of records: {df_test.shape[0]:,}")