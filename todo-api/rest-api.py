#!flask/bin/python
import churnPredict as cp
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return cp.centralFlow()

if __name__ == '__main__':
    app.run(debug=True)
