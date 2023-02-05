import sqlite3;

conn = sqlite3.connect('database.db')

sql_query = '''DROP TABLE IF EXISTS classes;

CREATE TABLE classes (cls_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        course_id INTEGER NOT NULL,
                        room_no TEXT NOT NULL,
                        floor TEXT,
                        date TEXT NOT NULL,
                        start_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (course_id) REFERENCES courses (c_id)
                        );'''

conn.executescript(sql_query)

#cur = conn.cursor()
# new_name = 'Adv. Mathematics'
# id = 6
#conn.execute("INSERT INTO classes (course_id, room_no, floor, date, start_time, end_time) VALUES (?,?,?,?,?,?)", ('3','123','1 O.G.','4/2/2023','13:00','15:00'))

conn.commit()
conn.close()