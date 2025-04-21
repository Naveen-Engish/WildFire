from flask import Flask, render_template, Response, redirect, url_for, request
import cv2
from ultralytics import YOLO
import threading
import sqlite3
from datetime import datetime
import os
import uuid
from flask import jsonify
from flask import session

app = Flask(__name__)
app.secret_key = 'your_secret_key' 
# Initialize database
conn = sqlite3.connect('wildfire_reports.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS reports
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              severity TEXT,
              location TEXT,
              event_time DATETIME,
              image_path TEXT,
              submitted_at DATETIME)''')
conn.commit()
c.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)''')
conn.commit()

# Detection system setup
model = YOLO('best.pt')
class_names = ['fire']
latest_detection = None
detection_lock = threading.Lock()
# Add global variables
camera = None
camera_enabled = True
camera_lock = threading.Lock()

# Initialize camera on startup
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



def detect_wildfire():
    global latest_detection
    while True:
        with camera_lock:
            if not camera_enabled or camera is None or not camera.isOpened():
                break
        success, frame = camera.read()
        if not success:
            break

        results = model(frame)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes):
                if score >= 0.8:  # Changed to 60% confidence threshold
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
                    filepath = os.path.join('static', 'detections', filename)

                    if not os.path.exists('static/detections'):
                        os.makedirs('static/detections')

                    cv2.imwrite(filepath, frame)

                    with detection_lock:
                        latest_detection = {
                            'image_path': filepath,
                            'timestamp': datetime.now().isoformat()
                        }

def generate_frames():
    while True:
        with camera_lock:
            if not camera_enabled or camera is None or not camera.isOpened():
                break
        success, frame = camera.read()
        if not success:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/')
def index():
    c.execute("SELECT id, severity, location, event_time, image_path, submitted_at FROM reports ORDER BY submitted_at DESC")
    reports = c.fetchall()
    return render_template('index.html', reports=reports, username=session.get('username'))



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_detection')
def check_detection():
    global latest_detection
    with detection_lock:
        if latest_detection:
            detection = latest_detection.copy()
            latest_detection = None  # Clear after retrieval
            return {'detected': True, 'image': detection['image_path']}
        return {'detected': False}

@app.route('/report', methods=['GET'])
def report_form():
    image_path = request.args.get('image')
    return render_template('report.html', 
                         image_path=image_path,
                         default_time=datetime.now().isoformat(timespec='minutes'))

@app.route('/submit_report', methods=['POST'])
def submit_report():
    # Save report to database
    c.execute('''INSERT INTO reports 
               (severity, location, event_time, image_path, submitted_at)
               VALUES (?, ?, ?, ?, ?)''',
               (request.form['severity'],
                request.form['location'],
                request.form['event_time'],
                request.form['image_path'],
                datetime.now().isoformat()))
    conn.commit()
    return redirect(url_for('index'))

@app.route('/retry', methods=['POST'])
def retry():
    image_path = request.form.get('image_path')
    try:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
            print(f"[INFO] Deleted false positive image: {image_path}")
        return jsonify({'success': True})
    except Exception as e:
        print(f"[ERROR] Failed to delete image: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        if user:
            session['username'] = username
            return redirect(url_for('index'))
        return "Invalid credentials"
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists"
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/delete_report', methods=['POST'])
def delete_report():
    if 'username' in session:
        report_id = request.form.get('report_id')
        c.execute("SELECT image_path FROM reports WHERE id=?", (report_id,))
        row = c.fetchone()
        if row:
            try:
                os.remove(row[0])
            except:
                pass
        c.execute("DELETE FROM reports WHERE id=?", (report_id,))
        conn.commit()
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_enabled, camera
    with camera_lock:
        camera_enabled = not camera_enabled
        if camera_enabled:
            if camera is None or not camera.isOpened():
                camera = cv2.VideoCapture(0)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Restart detection thread
            detection_thread = threading.Thread(target=detect_wildfire)
            detection_thread.daemon = True
            detection_thread.start()
        else:
            if camera is not None:
                camera.release()
        return jsonify({'success': True, 'camera_enabled': camera_enabled})

if __name__ == '__main__':
    detection_thread = threading.Thread(target=detect_wildfire)
    detection_thread.daemon = True
    detection_thread.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
