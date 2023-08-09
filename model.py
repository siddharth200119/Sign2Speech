from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import glob
from ast import literal_eval

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(101, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#reading data

path = r'Dataset_csv'
df = pd.concat(map(pd.read_csv, glob.glob(path + "/*.csv")))
df.head()

y = df.gesture
x = df.drop('gesture', axis = 1)

input = {
        "pose_points": [],
        "face_points": [],
        "lh_points": [],
        "rh_points": [],
        "pose_angles": [],
        "face_angles": [],
        "lh_angles": [],
        "rh_angles": []
    }

for index, row in x.iterrows():
        input['pose_angles'].append(row['pose_angles'])

print((type(input['pose_angles'])))

# X_train,X_test,y_train,y_test=train_test_split(input,y,test_size=0.05)

# model.fit(X_train, y_train, epochs=2000)

# model.save('model.h5')