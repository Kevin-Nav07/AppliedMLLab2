from flask import Flask, request, jsonify
from transformers import pipeline
import os
from dotenv import load_dotenv
load_dotenv()##loads name-value from .env into the os environment
##this entire file defines our API we create that connects to the model
MODEL_NAME = os.getenv("MODEL_NAME")##get env
##get port environment variable from operating system
PORT = int(os.getenv("PORT"))
## threshold constant we use to classify toxic vs non-toxic based off toxic score
THRESHOLD = float(os.getenv("THRESHOLD"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN")

##this line crates the flask application
app = Flask(__name__)

## Load the model ONCE at startup (fast requests later)
##once start up we load the model once instead of at eveyr API request
clf = None

def get_classifier():##lazy loading model
    global clf##consideronly the global variable

    if clf is None:##if no model is loaded, load the model
        clf = pipeline("text-classification",
            model=MODEL_NAME,
            token=HF_TOKEN if HF_TOKEN else None)
    return clf


@app.get("/health")##a get API endpoint for our api
def health():
    """
    Simple health check endpoint.
    Why: lets you quickly confirm the server is running.
    """
    return jsonify({"server response": "ok",
                    "threshold":THRESHOLD,
                    "model_name": MODEL_NAME})


@app.post("/predict")## post request endpoint for predicting a text
def predict():
    """
    REST endpoint that:
    - Reads JSON input: {"text": "..."}
    - Runs the model
    - Returns JSON output
    """

    ##Safely parse JSON (silent=True prevents Flask from throwing an HTML error page)
    data = request.get_json(silent=True)
    ##If body is empty or not JSON, data becomes None. We'll normalize to dict.
    if data is None:
        data = {}
    
    ##use the get method to retrieve the "text" attribute's value in the json
    text = data.get("text")

    ##Validate the input
    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "Invalid request. Send JSON like: {\"text\": \"your text here\"}"
        }), 400
    
        

    #running inference if the data is fine
    # pipeline returns a list of predictions; for a single input, take index 0
    classifier = get_classifier()
    raw = classifier(text)[0] 
    ##retrieve the score of toxicity because that is what we care about
    toxic_score = float(raw["score"])

    ##now conver the toxicity score to our binary decision of toxic or not
    label = "toxic" if toxic_score >= THRESHOLD else "non-toxic"

    ## Return a clean JSON response with our new custom threshold
    return jsonify({
        "label": label,
        "toxic_score": toxic_score,
        "threshold": THRESHOLD,
        "model_name":MODEL_NAME
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=FLASK_DEBUG)