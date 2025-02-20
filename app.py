from flask import Flask, request, render_template
import joblib
import re
import string

# Load models and vectorizer
import os
vectorizer_path = os.path.join(os.path.dirname(__file__), "model/vectorizer.pkl")
vectorizer = joblib.load(vectorizer_path)
nb_model = joblib.load("model/nb_model.pkl")
svm_model = joblib.load("model/svm_model.pkl")
rf_model = joblib.load("model/rf_model.pkl")

app = Flask(__name__)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"https?://\S+", "", text)  # Remove links
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return text

# Prediction function
def predict_sentiment(text, model):
    text = preprocess_text(text)
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return "Positive" if prediction[0] == 1 else "Negative"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["comment"]
        model_choice = request.form["model"]

        if model_choice == "Naive Bayes":
            model = nb_model
        elif model_choice == "SVM":
            model = svm_model
        elif model_choice == "Random Forest":
            model = rf_model
        else:
            return render_template("index.html", prediction="Error: Model not selected")

        result = predict_sentiment(user_input, model)
        return render_template("index.html", prediction=result, comment=user_input)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
