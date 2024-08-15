import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    data = pd.read_csv("QB_Data.csv")
    X = data.drop(columns=['Name','College','wAV'])
    y = data['wAV']
    z = [21,830,575,8123,12,85,1,80,136,75,215,27]
    #z_array = np.array(z).reshape(1, -1)
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)

    #model = tf.keras.Sequential([
    #    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    #    tf.keras.layers.Dense(32, activation='relu'),
    #    tf.keras.layers.Dense(1)  # Output layer with one neuron for regression
    #])

    #model.compile(optimizer='adam', loss='mean_squared_error')
    #model.fit(X_scaled, y, epochs=50, batch_size=32, verbose=1)
    #predictions = model.predict(X_scaled)
    #mse = mean_squared_error(y, predictions)
    #print("Mean Squared Error:", mse)

    model = DecisionTreeRegressor()
    model.fit(X, y)
    z_array = np.array(z).reshape(1, -1)
    predicted_impact = model.predict(z_array)
    print("Predicted impact:", predicted_impact)

    #predictions = model.predict(X)
    #mse = mean_squared_error(y, predictions)
    #print("Mean Squared Error:", mse)


