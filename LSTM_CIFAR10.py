# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:22:33 2024
LSTM - CIFAR10

@author: faranak
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Flatten, Reshape
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Filter the dataset to only include one class (e.g., class '0' for normal data)
normal_class = 0
normal_data = x_train[y_train.flatten() == normal_class]
normal_data = normal_data.astype('float32') / 255.0

# Add some anomalies (other classes)
anomaly_data = x_train[y_train.flatten() != normal_class]
anomaly_data = anomaly_data[:500]  # Select 500 anomalies
anomaly_data = anomaly_data.astype('float32') / 255.0

# Combine normal data and anomaly data
data = np.vstack([normal_data, anomaly_data])

# Create ground truth labels
ground_truth = np.hstack([np.zeros(normal_data.shape[0]), np.ones(anomaly_data.shape[0])])

# Standardize the data
data_flat = data.reshape(data.shape[0], -1)  # Flatten for clustering
scaler = StandardScaler()
data_flat = scaler.fit_transform(data_flat)
data = data_flat.reshape(data.shape[0], 32, 32 * 3)  # Reshape to (samples, time_steps, features)

# Clustering clients based on data similarities
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters)
clusters = kmeans.fit_predict(data_flat)

# Define an LSTM autoencoder model
def create_lstm_autoencoder(input_shape):
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(RepeatVector(input_shape[0]))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(input_shape[1])))
    model.compile(optimizer='adam', loss='mse')
    return model

# Initialize models for each cluster
input_shape = data.shape[1:]  # (time_steps, features)
models = [create_lstm_autoencoder(input_shape) for _ in range(num_clusters)]

# Simulate local training on clients
epochs = 10
batch_size = 32

for epoch in range(epochs):
    for cluster_id in range(num_clusters):
        cluster_data = data[clusters == cluster_id]
        models[cluster_id].fit(cluster_data, cluster_data, epochs=1, batch_size=batch_size, verbose=0)

    # Simulate aggregation by cluster leaders and global aggregation
    global_weights = np.mean([model.get_weights() for model in models], axis=0)
    for model in models:
        model.set_weights(global_weights)

# Anomaly detection
anomalies = []

# Initialize predictions
predictions = np.zeros(data.shape[0])

# Calculate reconstruction errors for all samples
reconstruction_errors = []
for sample in data:
    reconstruction = models[0].predict(sample.reshape(1, *input_shape))  # Use any model for prediction
    error = np.mean((sample - reconstruction) ** 2)
    reconstruction_errors.append(error)

# Determine optimal threshold using a range of potential thresholds
best_f1 = 0
best_threshold = 0
for threshold in np.linspace(min(reconstruction_errors), max(reconstruction_errors), num=100):
    temp_predictions = (np.array(reconstruction_errors) > threshold).astype(int)
    f1 = f1_score(ground_truth, temp_predictions)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Use the best threshold for final predictions
predictions = (np.array(reconstruction_errors) > best_threshold).astype(int)

# Calculate metrics
accuracy = accuracy_score(ground_truth, predictions)
precision = precision_score(ground_truth, predictions)
recall = recall_score(ground_truth, predictions)
f1 = f1_score(ground_truth, predictions)

print(f"Optimal Threshold: {best_threshold}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-1 Score: {f1}")

# Plot some sample anomalies
fig, axes = plt.subplots(1, 10, figsize=(20, 2))
anomaly_samples = data[predictions == 1]
for i, ax in enumerate(axes):
    if i < len(anomaly_samples):
        ax.imshow(anomaly_samples[i].reshape(32, 32, 3), cmap='gray')
    ax.axis('off')
plt.suptitle("Sample Anomalies Detected")
plt.show()
