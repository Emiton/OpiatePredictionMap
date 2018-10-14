import os

import datetime

from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions
from werkzeug.security import check_password_hash, generate_password_hash
from flask_session import Session
from opioidmap import cool
from predictions import brenno

# Configure application
app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
#db = SQL("sqlite:///finance.db")

@app.route("/")
def home():
    person = cool()
    return render_template("index.html", name = person )



@app.route("/machineLearning")
def learn():
    thing = brenno()
    return render_template("machineLearning.html", name = thing)


@app.route("/plotData")
def plot():
    thing = 'PLOT'
    return render_template("plotData.html", name = thing)


@app.route("/heatMap")
def heat():
    thing = "MAP"
    return render_template("heatMap.html", name = thing)