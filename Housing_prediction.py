import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("housing_data.csv")
print("Dataset Loaded Successfully\n")
print(data)

# Features and target
X = data[['Area', 'Rooms', 'Age', 'Distance']]
y = data['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
data['Predicted_Price'] = model.predict(X)

# Plot connected graph
plt.plot(data['Year'], data['Price'], marker='o', label='Actual HPI')
plt.plot(data['Year'], data['Predicted_Price'], marker='o', linestyle='--', label='Predicted HPI')

plt.xlabel("Year")
plt.ylabel("Housing Price Index")
plt.title("Housing Price Index Prediction")
plt.legend()
plt.grid(True)
plt.show()
