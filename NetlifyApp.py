from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('random_forest_model.joblib')

@app.route('/')
def home():
    return "Hello, this is your model deployment!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Assuming your model takes input and returns predictions
    input_data = data['input']
    prediction = model.predict([input_data])

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
