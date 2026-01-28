from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

