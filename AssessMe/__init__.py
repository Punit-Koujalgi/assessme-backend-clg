from flask import Flask
from flask_cors import CORS

application=app=Flask(__name__)

CORS(application)

from AssessMe import routes