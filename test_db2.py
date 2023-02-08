import sqlite3;

conn = sqlite3.connect('database.db')

sql_query = '''DROP TABLE IF EXISTS students;

CREATE TABLE students (s_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        f_name TEXT NOT NULL,
                        l_name TEXT NOT NULL,
                        email_id TEXT NOT NULL,
                        dob TEXT,
                        gender TEXT,
                        address_line_1 TEXT,
                        address_line_2 TEXT,
                        address_city TEXT NOT NULL,
                        address_state TEXT NOT NULL,
                        address_country TEXT NOT NULL,
                        address_plz TEXT NOT NULL,
                        image_link TEXT NOT NULL
                        );'''

# sql_query = '''DROP TABLE IF EXISTS attendences;

# CREATE TABLE attendences (a_id INTEGER PRIMARY KEY AUTOINCREMENT,
#                             s_id INTEGER,
#                             f_name TEXT,
#                             class_id INTEGER,
#                             course_id INTEGER,
#                             date DATE,
#                             time TIMESTAMP,
#                             FOREIGN KEY(s_id) REFERENCES classes(cls_id));
# '''

conn.executescript(sql_query)

#cur = conn.cursor()
# new_name = 'Adv. Mathematics'
# id = 6
#conn.execute("UPDATE sqlite_sequence SET seq = 1000 WHERE NAME = 'students';")

#conn.execute("INSERT INTO students (f_name, l_name, email_id, dob, gender, address_line_1, address_line_2, address_city, address_state, address_country, address_plz) VALUES (?,?,?,?,?,?,?,?,?,?,?)", 
#             ('Chaitanya','Manjrekar','cm@gmail.com','09/02/2001','M','Luebecker Strasse 5','','Eschborn','Hesse','Germany','65760'))

#conn.execute("DELETE FROM students WHERE s_id BETWEEN 5 AND 8;")
#conn.execute("DELETE FROM attendences;")

# cur = conn.cursor()
# count = cur.execute("SELECT * FROM attendences;").fetchall()

# print(count)

conn.commit()
conn.close()