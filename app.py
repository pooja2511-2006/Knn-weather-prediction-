import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Page setup
st.set_page_config(page_title="KNN Weather Classifier")
st.title("KNN Weather Classification")

# Training data
X = np.array([[50, 70], [25, 80], [27, 60], [31, 65], [23, 85], [20, 75]])
y = np.array([0, 1, 0, 0, 1, 1])  # 0 = Sunny, 1 = Rainy
label_map = {0: "Sunny", 1: "Rainy"}

# Sidebar inputs
st.sidebar.header("Input Features")
temp = st.sidebar.slider("Temperature", 10, 60, 26)
hum = st.sidebar.slider("Humidity", 50, 95, 78)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Prediction
new_data = np.array([[temp, hum]])
prediction = knn.predict(new_data)[0]
st.write(f"Predicted Weather: **{label_map[prediction]}**")

# Plot decision boundary
fig, ax = plt.subplots()

# Create mesh grid
x_min, x_max = 10, 60
y_min, y_max = 50, 95
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 1))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Background decision regions
ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# Training points
ax.scatter(X[y==0, 0], X[y==0, 1], color='orange', label='Sunny', s=100, edgecolor='k')
ax.scatter(X[y==1, 0], X[y==1, 1], color='blue', label='Rainy', s=100, edgecolor='k')

# New prediction point
ax.scatter(temp, hum, color='red' if prediction==1 else 'orange', marker='*',
           s=300, edgecolor='black', label=f'New Day: {label_map[prediction]}')

# Labels and styling
ax.set_xlabel('Temperature')
ax.set_ylabel('Humidity')
ax.set_title('KNN Weather Classification with Decision Boundary')
ax.legend()
ax.grid(True)

st.pyplot(fig)
