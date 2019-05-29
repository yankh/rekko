#!/usr/bin/python3
from flask import Flask, render_template, request, url_for, redirect, jsonify, send_from_directory

app = Flask(__name__)

# Website main page
@app.route('/')
def landing():
    return render_template('admin.html')



    
if __name__ == '__main__':
    app.run(debug=True, host= '0.0.0.0')
