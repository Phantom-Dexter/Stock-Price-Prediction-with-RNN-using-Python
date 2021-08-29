import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


path = str(input("Enter your file path:\n"))
df = pd.read_csv(path)
df = df[str(input("Enter column for Prediction:\n"))]


def plot_series(data):
    for df in data:
        plt.plot(df)
    plt.show()

train = df[:int(0.8 * len(df))]
test = df[int(0.8 * len(df)):]

# Arraning the data that we have sto suit the model that we want (Reshaping)
def arrange(data, window=10):
    x = []
    y = []
    
    #Loop through the data
    for i, val in enumerate(data):
        if i < window:
            #Continue if the number of past records is not enough
            continue
        x.append(data[i-window: i-1].values.reshape(-1, 1))
        y.append(data[i:i+1].values.reshape(-1, 1))
        
    x = np.asarray(x)
    y = np.asarray(y)
    
    return x, y

x_train, y_train = arrange(train, 15)
print("x-shape is: {} and y-shape is: {}".format(x_train.shape, y_train.shape))


#------------------ BUILDING THE MODEL ------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Input


Network = Sequential()
Network.add(SimpleRNN(3, activation='relu', input_shape=x_train.shape[1:]))
Network.add(Dense(1, activation= 'relu'))
Network.compile(loss ='mean_squared_error', optimizer = 'adam', metrics=['mse'])

Network.fit(x_train, y_train, epochs=10)
x_test, y_test = arrange(test, 15)
pred = Network.predict(x_test)
plot_series([pred[:, 0], y_test[:,:,0]])






