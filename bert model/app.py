from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

sentiment_analyzer = pipeline("sentiment-analysis")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET','POST'])
def analyze():
    if request.method == "POST":
        input_text = request.form["inputtext_"]
        sentiment_result = sentiment_analyzer(input_text)[0]
        print(input_text)


    return render_template("output.html", data={"sentiment": sentiment_result["label"],
                                                "score": sentiment_result["score"],
                                                "input_text": input_text})


if __name__ == "__main__":
    app.run(port=8000)
