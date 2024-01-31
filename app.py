import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    
    prediction = model.predict(final_features)
    
    return render_template('index.html',prediction_text='Predicted class: {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)