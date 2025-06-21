from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pickled model
with open('Model/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input from form
        features = [float(x) for x in request.form.values()]
        input_array = np.array([features])

        # Make prediction
        prediction = model.predict(input_array)
        result = "Anemic" if prediction[0] == 1 else "Not Anemic"

        return render_template('index.html', prediction_text=f'Result: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
