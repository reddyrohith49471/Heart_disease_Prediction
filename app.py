import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

LR = pickle.load(open('LR.pkl','rb'))
scaling = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json()['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaling.transform(np.array(list(data.values())).reshape(1,-1))
    output = LR.predict(new_data)
    print(output[0])
    return jsonify({'prediction': int(output[0])})

if __name__ == "__main__":
    app.run(debug=True)
