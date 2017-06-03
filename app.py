import numpy as np
from flask import Flask, abort, jsonify, request
import cPickle as pickle 

random_forest_model = pickle.load(open("rfc.pkl","rb"))

app = Flask(__name__)

@app.route('/predict_api', methods=['POST'])
def predict():
     # Error checking
     data = request.get_json(force=True)

     # Convert JSON to numpy array
     predict_request = [data['sl'],data['sw'],data['pl'],data['pw']]
     predict_request = np.array(predict_request)

     # Predict using the random forest model
     y = random_forest_model.predict(predict_request)

     # Return prediction
     output = [y[0]]
     return jsonify(results=output)

if __name__ == '__main__':
     app.run(port = 9000, debug = True)
