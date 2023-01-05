from flask import Flask, jsonify, request
from prediction_engine import PredictionModel

app = Flask(__name__)
model = PredictionModel()


@app.post('/')
def handle_post():
    if request.is_json:
        title = request.get_json()["heading"]
        legitimacy = model.predict(title)
        response = {
            "heading": title,
            "legitimacy": legitimacy
        }
        return jsonify(response)
    else:
        return {"error": "Request must be JSON"}, 415


if __name__ == '__main__':
    app.run()
