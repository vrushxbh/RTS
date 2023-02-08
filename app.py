from datetime import datetime, timezone, date
import re
from flask import Flask, Response, flash, redirect, render_template, request, session, url_for
import os
import cv2
import psycopg2
import psycopg2.extras
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import face_recognition
import numpy as np
import joblib
import sqlite3

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
face_path = os.getenv("FACE_DIR")


app = Flask(__name__, template_folder=template_dir)
app.secret_key = os.getenv("SECRET_KEY")

connection = psycopg2.connect(url)

face_cascade = cv2.CascadeClassifier()

# Load the pretrained model
#face_cascade.load(cv2.samples.findFile("static/haarcascade_frontalface_alt.xml"))
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def gen_frames():
    video = cv2.VideoCapture(0)
    while True:
        success, frame = video.read()
        if success != True:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield(b'--frame\r\n' 
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        if cv2.waitKey(1)==27:
            break
    #video.release()

def mark_attendence(id, class_id, course_id):
    # att_path = os.getenv("ATTENDENCE_DIR")
    # att_path = att_path + "/" + datetime.now().strftime('%B')+str(datetime.now().year)
    # if os.path.exists(att_path) == True:
    #     print('Directory exists')
    # else:
    #     os.makedirs(att_path)
    #     print('Directory created!')

    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT COUNT(*) FROM attendences WHERE s_id = "+id+" AND date = DATE('now')"
    rec_exists = True if (cursor.execute(query).fetchone())[0] > 0 else False
    if rec_exists == True:
        print("Record already exists for today.")
    else:
        #cursor.execute("SELECT s.s_id, s.f_name,")
        conn.execute("INSERT INTO attendences(s_id, f_name, class_id, course_id, date, time) VALUES (?,?,?,?,?,?)",
                     (id, "Test", class_id, course_id, date.today().strftime('%d/%m/%Y'),datetime.now().strftime('%H:%M')))
    conn.commit()
    conn.close()

def calculate_histogram(image):
    histogram = [0] * 3
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist *= 255.0 / hist.max()
        histogram[i] = hist
    return np.array(histogram)

def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# def recognize(input_image):
#     known_face_path = os.getenv("KNOWN_FACE_DIR")
#     images = []
#     classNames = []
#     imgList = os.listdir(known_face_path)
#     for image in imgList:
#         if (image.endswith(".jpg")):
#             currImg = cv2.imread(f'{known_face_path}/{image}')
#             images.append(currImg)
#             classNames.append(os.path.splitext(image))
#             encodeListKnown = find_encodings(images)
        
#         else:
#             print("Please first add faces!")

#     print(classNames)


    test_img = face_recognition.load_image_file(input_image)
    test_img = cv2.cvtColor(test_img)

def gen():
    no_of_imgs = int(os.getenv("IMAGES_TO_CAPTURE"))
    if os.path.exists(face_path) == True:
        print('Directory exists')
    else:
        os.makedirs(face_path)
        print('Directory created!')

    known_face_path = os.getenv("KNOWN_FACE_DIR")
    if os.path.exists(known_face_path) == True:
        print('Directory exists')
    else:
        os.makedirs(known_face_path)
        print('Directory created!')

    """ model_path = os.getenv("MODEL_PATH")
    if os.path.exists(model_path) == True:
        print('Directory exists')
    else:
        print('Please load/specify a trained model') """
    
    images = []
    classNames = []
    currAttendence = []
    imgList = os.listdir(known_face_path)
    #model = joblib.load(model_path)
    #print("Model loaded successfully")

    for image in imgList:
        if (image.endswith(".jpg")):
            currImg = cv2.imread(f'{known_face_path}/{image}')
            images.append(currImg)
            classNames.append(os.path.splitext(image)[0])
            encodeListKnown = find_encodings(images)
        
        else:
            print("Please first add faces!")

    print(classNames)
    cap = cv2.VideoCapture(0)
    i=0
    sample_number = 1
    count = 0
    #measures = np.zeros(sample_number, dtype=np.float)
    while True:
        ret,frame = cap.read()
        #measures[count%sample_number]=0
        #height, weight = frame.shape[:2]
        #if extract_faces(frame)!=():
        
            #cv2.putText(frame,'test',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
        frameS = cv2.resize(frame, (0,0), None, 0.25, 0.25)
        frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)
        faceCurrFrame = face_recognition.face_locations(frameS)
        encodeCurrFrame = face_recognition.face_encodings(frameS, faceCurrFrame)
        #cv2.imwrite(face_path+'/test'+str(i)+'.jpg', frame)
            
        for encodedFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodedFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodedFace)
            print(faceDis)
            matchedInd = np.argmin(faceDis)

            if matches[matchedInd]:
                name = classNames[matchedInd].upper()
                if name not in currAttendence:
                    currAttendence.append(name)
                else:
                    pass
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 2)
                cv2.rectangle(frame, (x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED)
                cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

        cv2.imshow('Webcam',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print(currAttendence)
    return currAttendence
    # for i in currAttendence:
    #     mark_attendence(i)
    # currAttendence.clear()

@app.route('/')
def home():
    if 'loggedin' in session:
        conn = get_db_connection()
        courses = conn.execute("SELECT * FROM courses").fetchall()
        classes = conn.execute("SELECT * FROM classes").fetchall()
        conn.close()
        print(len(courses))
        print(len(classes))
        img_list = ['https://img.freepik.com/free-vector/typing-concept-illustration_114360-363.jpg?w=2000v',
'https://cdni.iconscout.com/illustration/premium/thumb/coding-study-4024615-3328754.png',
'https://media.istockphoto.com/id/1241710244/vector/working-at-home-vector-flat-style-illustration-online-career-coworking-space-illustration.jpg?s=612x612&w=0&k=20&c=U34U9zhLBWDEbfPmgmlnFJiP-EuWu7MEUCxUls_BnKU=',
'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS1Y4-E5TjYjx-WZPSDHmNUZydfxvI9F3PTFP1FNJkjuAhvhYKzBb9CgZD-2fnfsKQDtcY&usqp=CAU',
'https://static.vecteezy.com/system/resources/thumbnails/000/116/552/small/free-flat-line-business-and-marketing-vector-illustration.jpg',
'https://static.vecteezy.com/system/resources/thumbnails/000/152/199/small/linear-network-technology-icons.jpg',
'https://static.vecteezy.com/system/resources/thumbnails/000/140/421/small/free-vector-illustration-of-web-learning-concept.jpg']
        
        return render_template('home.html', username=session['username'], courses=courses, classes=classes, img_list=img_list, i=0)
    
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

@app.route('/student', methods=["GET","POST"])
def student():
    cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

    if 'loggedin' in session:
        cursor.execute('SELECT * FROM emp WHERE e_id = %s', [(session['id'])])
        account = cursor.fetchone()

        conn = get_db_connection()
        cur = conn.cursor()
        students = cur.execute("SELECT * FROM students").fetchall()
        conn.commit()
        conn.close()

        return render_template('student.html', account=account, students=students)
    
    return redirect(url_for('login'))

@app.route('/create_student', methods=["GET", "POST"])
def create_student():
    if request.method == "POST":
        f_name = request.form['f_name']
        l_name = request.form['l_name']
        email_id = request.form['email_id']
        dob = request.form['dob']
        gender = request.form.get('gender')
        add_1 = request.form['address_line_1']
        add_2 = request.form.get('address_line_2', False)
        city = request.form['city']
        state = request.form['state']
        country = request.form['country']
        plz = request.form['plz']
        fileobj = request.files['image']
        file_extensions =  ["JPG"]
        uploaded_file_extension = fileobj.filename.split(".")[1]
        #validating file extension
        if(uploaded_file_extension.upper() in file_extensions):

            conn = get_db_connection()
            cur = conn.cursor()
            query = "SELECT * FROM students WHERE f_name='"+f_name+"' AND l_name='"+l_name+"' AND email_id='"+str(email_id)+"';"
            # print(query)
            #valid = [0,0]
            valid = cur.execute(query).fetchall()
            if len(valid)!=0:
                flash("Student already exists.")
            else:
                try:
                    filename = cur.execute("SELECT MAX(s_id) FROM students").fetchone()
                    if filename['MAX(s_id)'] is None:
                        filename = '1.jpg'
                    else:
                        filename = str(filename['MAX(s_id)']+1)+'.jpg'
                    print(filename)
                    fileobj.filename = filename
                    destination_path= f"static/known_faces/"
                    fileobj.save(destination_path+filename)
                    #inserting data into table usercontent
                    conn.execute("INSERT INTO students (f_name, l_name, email_id, dob, gender, address_line_1, address_line_2, address_city, address_state, address_country, address_plz, image_link) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", 
                (f_name,l_name, email_id, dob, gender, add_1, add_2, city, state, country, plz,destination_path))
                    conn.commit()
                    conn.close()
                except sqlite3.Error as error:
                    #using flash function of flask to flash errors.
                    flash(f"{error}")
            return redirect(url_for('student'))
        else:
            flash("Only images are accepted")
    return redirect(url_for('student'))


@app.route('/attendence')
def attendence():
    cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
    if 'loggedin' in session:
        cursor.execute('SELECT * FROM emp WHERE e_id = %s', [(session['id'])])
        account = cursor.fetchone()
        query = "SELECT * FROM attendences;"
        conn = get_db_connection()
        cur = conn.cursor()
        attendences = cur.execute(query).fetchall()
        print(attendences)
        conn.commit()
        conn.close()

        return render_template('attendence.html', account=account, attendences=attendences)
    return redirect(url_for('login'))

@app.route('/video_feed/<int:class_id>/<int:course_id>')
def video(class_id, course_id):
    if 'loggedin' in session:
        #global video
    #return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        currAttendence = gen()
        print(currAttendence)
        for i in currAttendence:
            mark_attendence(i, class_id, course_id)
        currAttendence.clear()
        #return render_template('attendence.html', username=session['username'])
        return redirect(url_for('attendence'))
    return redirect(url_for('login'))
    
    #return Response('Closed')
    #return redirect(url_for('attendence'))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


@app.route('/create_course', methods=["POST"])
def create_course():
    if request.method == 'POST':
        class_name = request.form.get("class_name", False)
        conn = get_db_connection()
        conn.execute('INSERT INTO courses (c_name) VALUES (?)',(class_name, ))
        conn.commit()
        conn.close()
        flash('Course added successfully.')
        return redirect(url_for('home'))

@app.route('/delete_course', methods=["POST"])
def delete_course():
    if request.method == "POST":
        conn = get_db_connection()
        c_id = request.form.get("c_id", False)
        conn.execute('DELETE FROM courses WHERE c_id = (?)', (c_id,))
        conn.commit()
        conn.close()
        flash("Course deleted successfully.")
        return redirect(url_for('home'))


@app.route('/update_course', methods = ["GET", "POST"])
def update_course():
    if request.method == "POST":
        conn = get_db_connection()
        c_id = request.form.get("c_id", False)
        upd_name = request.form['course_name']
        conn.execute('UPDATE courses SET c_name = (?) WHERE c_id = (?)', (upd_name, c_id))
        conn.commit()
        conn.close()
        flash("Course update successfully.")
        return redirect(url_for('home'))

#------------CLASSES-------------------
@app.route('/create_class', methods=["GET", "POST"])
def create_class():
    if request.method == "POST":
        course_id = int(request.form.get("c_id", False))
        room_no = request.form['room_no']
        floor = request.form['floor']
        date = request.form['date']
        start_time = request.form['start_time']
        end_time = request.form['end_time']
        conn = get_db_connection()
        conn.execute("INSERT INTO classes(course_id, room_no, floor, date, start_time, end_time) VALUES (?,?,?,?,?,?)",(course_id, room_no, floor, date, start_time, end_time))
        conn.commit()
        conn.close()
        flash("Class created successfully.")
        return redirect(url_for('home'))

@app.route('/update_class', methods=["GET", "POST"])
def update_class():
    if request.method == "POST":
        class_id = int(request.form.get("cls_id", False))
        room_no = request.form['room_no']
        floor = request.form['floor']
        start_time = request.form['start_time']
        end_time = request.form['end_time']
        conn = get_db_connection()
        conn.execute("UPDATE classes SET room_no=(?), floor=(?), start_time=(?), end_time=(?) WHERE cls_id=(?)",(room_no, floor, start_time, end_time, class_id))
        conn.commit()
        conn.close()
        flash("Class update successfully.")
        return redirect(url_for('home'))

@app.route('/delete_class', methods=["GET", "POST"])
def delete_class():
    if request.method == "POST":
        class_id = request.form.get("cls_id", False)
        conn = get_db_connection()
        conn.execute("DELETE FROM classes WHERE cls_id=(?)", (class_id,))
        conn.commit()
        conn.close()
        flash("Class deleted successfully.")
        return redirect(url_for('home'))

#---------------Attendence-------------------------------------
@app.route('/delete_attendence', methods=["GET","POST"])
def delete_attendence():
    if request.method == "POST":
        att_id = request.form.get("a_id", False)
        conn = get_db_connection()
        conn.execute("DELETE FROM attendences WHERE a_id=(?)", (att_id,))
        conn.commit()
        conn.close()
        flash("Attendence deleted successfully.")
        return redirect(url_for('attendence'))

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