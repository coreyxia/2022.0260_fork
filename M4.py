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


model='M4'
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Random Forest without sampling
# Scale the data to be between -1 and 1
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# create regressor object
regressor = RandomForestRegressor(n_estimators = 30, random_state = 0)
# fit the regressor with x and y data
reg=regressor.fit(X_train, y_train)

test_predict=reg.predict(X_test)
dataset = pd.DataFrame({model: test_predict})
dataset.to_csv(path4)

# # Establish model
# model = RandomForestRegressor(n_jobs=-1)
# # Try different numbers of n_estimators - this will take a minute or so
# estimators = np.arange(10, 100, 10)
# scores = []
# for n in estimators:
#     model.set_params(n_estimators=n)
#     model.fit(X_train, y_train)
#     scores.append(model.score(X_test, y_test))
# plt.title("Effect of n_estimators")
# plt.xlabel("n_estimator")
# plt.ylabel("score")
# plt.plot(estimators, scores)