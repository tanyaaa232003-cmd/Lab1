#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# streamlit_fashion_mnist.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import accuracy_score
import joblib  # use joblib to load scikit-learn models

# ---------------- Load Fashion MNIST ---------------- #
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_test_flat = X_test.reshape(-1, 784).astype("float32") / 255.0

# Class names
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

# ---------------- Load trained model ---------------- #
best_mlp_model = joblib.load("best_mlp_model.pkl")

# ---------------- Streamlit UI ---------------- #
st.title("Fashion MNIST Random Prediction")

# Pick 10 truly random test samples
idxs = np.random.choice(len(X_test), size=10, replace=False)
samples_flat = X_test_flat[idxs]

# Predict using the loaded model
preds_out = best_mlp_model.predict(samples_flat)

# Handle probabilities or direct class labels
if preds_out.ndim > 1:
    preds = np.argmax(preds_out, axis=1)
else:
    preds = preds_out

# Display images with predictions and true labels
st.subheader("Randomly Selected Test Images with Predictions")
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, (idx, pred) in enumerate(zip(idxs, preds)):
    ax = axes[i//5, i%5]
    ax.imshow(X_test[idx], cmap="gray")
    ax.set_title(f"Pred: {class_names[pred]}\nTrue: {class_names[y_test[idx]]}")
    ax.axis("off")
st.pyplot(fig)

# Display accuracy on these 10 samples
acc = accuracy_score(y_test[idxs], preds)
st.subheader(f"Accuracy on these 10 random samples: {acc:.2f}")

