import joblib

# Load the saved model
model = joblib.load("crop_model.pkl")

# Get user input
print("Enter the following soil and weather details:")
N = float(input("Nitrogen (N): "))
P = float(input("Phosphorus (P): "))
K = float(input("Potassium (K): "))
temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
ph = float(input("pH value: "))
rainfall = float(input("Rainfall (mm): "))

# Prepare input data for prediction
input_data = [[N, P, K, temperature, humidity, ph, rainfall]]

# Make prediction
prediction = model.predict(input_data)

print("\n✅ Recommended Crop for given conditions:", prediction[0])

