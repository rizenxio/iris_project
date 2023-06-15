from flask import Flask, jsonify, request, render_template
import pickle

# Muat model dari file
with open('model_knn.pkl', 'rb') as file:
    knn = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Menggunakan request.get_json() untuk mendapatkan data JSON dari permintaan
    
    # Pastikan bahwa data yang diterima adalah array 2 dimensi
    if isinstance(data, list) and all(isinstance(row, list) for row in data):
        predictions = knn.predict(data)
        response = {'predictions': predictions.tolist()}  # Membungkus hasil prediksi dalam dictionary
        return jsonify(response)

    response = {'error': 'Invalid data format'}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
