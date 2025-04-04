from flask import Flask, request, jsonify
import torch 
from clip import trained_model, predict_image  

app = Flask(__name__)
model = trained_model()

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json 
    prediction = predict_image(model, data) 
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
