#!/usr/bin/python3
from flask import Flask, render_template, request, url_for, redirect, jsonify, send_from_directory

app = Flask(__name__)

#Logout page displaying
@app.route('/')
def logout():
    return render_template('login.html')

# Login page displaying
@app.route('/', methods = ['POST'])
def login():
    if request.method == 'POST' and 'login' in request.form:
        if (request.form['password'] == "popo") and (request.form['username'] == "admin"):
            return render_template('admin.html')
        else:
            return render_template('login.html', wrong=True)


# #Setting
# @app.route('/')
# def setting():
#     return render_template('login.html')




if __name__ == '__main__':
    app.run(debug=True, host= '0.0.0.0')
