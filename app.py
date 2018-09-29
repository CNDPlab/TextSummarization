from flask import Flask
from flask import render_template, request

app = Flask(__name__)


@app.route('/')
def welcome():
    return render_template('base.html')

@app.route('/', methods=['POST'])
def process():
    text = request.form['text']
    processed = text.upper()

    return render_template("processed.html", result = processed)

if __name__ == '__main__':
    app.run(debug=True)
