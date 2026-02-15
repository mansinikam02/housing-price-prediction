import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

# Page title
st.title("🏠 Housing Price Index Prediction")

# Load dataset
data = pd.read_csv("housing_data.csv")

# Show dataset
st.subheader("Housing Dataset")
st.dataframe(data)

# Select features and target
X = data[['Area', 'Rooms', 'Age', 'Distance']]
y = data['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict prices (HPI)
data['Predicted_Price'] = model.predict(X)

# Plot connected graph
st.subheader("Housing Price Index Trend")
fig, ax = plt.subplots()

ax.plot(data['Year'], data['Price'],
        marker='o', label='Actual HPI')

ax.plot(data['Year'], data['Predicted_Price'],
        marker='o', linestyle='--', label='Predicted HPI')

ax.set_xlabel("Year")
ax.set_ylabel("Housing Price Index")
ax.set_title("Actual vs Predicted Housing Price Index")
ax.legend()
ax.grid(True)

# Display graph
st.pyplot(fig)
