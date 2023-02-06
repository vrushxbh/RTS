from datetime import datetime, timezone
import re
import sqlite3
from flask import Flask, flash, redirect, render_template, request, session, url_for
import os
import psycopg2
import psycopg2.extras
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

CREATE_EMPLOYEE_TABLE = (
    "CREATE TABLE IF NOT EXISTS emp (e_id SERIAL PRIMARY KEY, fullname TEXT, username TEXT, password TEXT, email TEXT, created_on TIMESTAMP);"
)

INSERT_EMPLOYEE_RETURN_ID = "INSERT INTO emp (username, password, created_on) VALUES (%s, %s, %s) RETURNING e_id;"

GLOBAL_EMP_AUTH = (
    "SELECT COUNT(username) AS valid FROM employee WHERE username=vrushxbh;"
)

load_dotenv()

template_dir = os.getenv("TEMPLATE_DIR")
url = os.getenv("DATABASE_URL")


app = Flask(__name__, template_folder=template_dir)
app.secret_key = os.getenv("SECRET_KEY")

connection = psycopg2.connect(url)

@app.route('/')
def home():
    if 'loggedin' in session:
        return render_template('home.html', username=session['username'])
    
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        fullname = request.form['fullname']
        username = request.form['username']
        password = request.form['password']
        _hashed_password = generate_password_hash(password)
        email = request.form['email']
        created_on = datetime.now(timezone.utc)

        cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(CREATE_EMPLOYEE_TABLE)
        cursor.execute('SELECT * FROM emp WHERE username = %s', (username,))
        account = cursor.fetchone()
        print(account)

        #If account already exists
        if account:
            flash('Account already exists')
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash('Username must contain only characters and numbers')
        else:
            cursor.execute("INSERT INTO emp (fullname, username, password, email, created_on) VALUES (%s,%s,%s,%s,%s)", (fullname, username, _hashed_password, email, created_on))
            connection.commit()
            flash('You have successfully registered!')
    elif request.method == 'POST':
        flash('Please fill out the form')

    return render_template('register.html')


@app.route('/login/', methods=['GET','POST'])
def login():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute('SELECT * FROM emp WHERE username = %s', (username,))
        account = cursor.fetchone()

        if account:
            password_rs = account['password']
            if check_password_hash(password_rs, password):
                session['loggedin'] = True
                session['id'] = account['e_id']
                session['username'] = account['username']

                return redirect(url_for('home'))
            else:
                flash('Incorrect username/password')
        else:
            flash('Incorrect username/password')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)

    return redirect(url_for('login'))


@app.route('/profile')
def profile():
    cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

    if 'loggedin' in session:
        cursor.execute('SELECT * FROM emp WHERE e_id = %s', [(session['id'])])
        account = cursor.fetchone()

        return render_template('profile.html', account=account)
    
    return redirect(url_for('login'))

@app.route('/recognize')
def recognize():
    cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

    if 'loggedin' in session:
        cursor.execute('SELECT * FROM emp WHERE e_id = %s', [(session['id'])])
        account = cursor.fetchone()

        return render_template('recognize.html', account=account)
    return redirect(url_for('login'))
"""
url = os.getenv("DATABASE_URL")
connection = psycopg2.connect(url)

@app.post("/api/emp")
def create_employee():
    data = request.get_json()
    username = data["username"]
    passwd = data["password"]
    try:
        created_at = datetime.strptime(data["created_at"], "%m-%d-%Y %H:%M:%S")
    except KeyError:
        created_at = datetime.now(timezone.utc)

    with connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_EMPLOYEE_TABLE)
            cursor.execute(INSERT_EMPLOYEE_RETURN_ID, (username, passwd, created_at,))
            emp_id = cursor.fetchone()[0]
    return {"emp_id": emp_id, "message": f"Employee {username} created successfully."}, 201

@app.get("/api/emp")
def get_employee():
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(GLOBAL_EMP_AUTH)
            valid_flag = cursor.fetchone()[0]
            if valid_flag == 1:
                msg = "Username exists"
            else:
                msg = "Invalid username"
    return {"message": msg}
    
"""    