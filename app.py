# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
from datetime import datetime
import math, threading, time, os, cv2
import numpy as np
from collections import deque
from keras.models import load_model
from ultralytics import YOLO
import queue

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///threat_recognition.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'  # Change to a secure secret in production

# Flask-Mail configuration (use environment variables for sensitive info in production)
app.config['MAIL_SERVER'] = 'smtp.sendgrid.net'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'apikey'  # This is literally the string "apikey"
app.config['MAIL_PASSWORD'] = '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'
app.config['MAIL_DEFAULT_SENDER'] = 'threatrec@gmail.com'

mail = Mail(app)
db = SQLAlchemy(app)

# ------------------------
# Flask-Login Setup
# ------------------------
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ------------------------
# Database Models
# ------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), default='user')  # "user" or "admin"
    contacts = db.relationship('Contact', backref='user', lazy=True)

class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(255), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    working = db.Column(db.Boolean, default=True)  # New column to track camera status

class Incident(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_latitude = db.Column(db.Float, nullable=False)
    user_longitude = db.Column(db.Float, nullable=False)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    violence_detected = db.Column(db.Boolean, default=False)  # Flag for violence detection
    weapon_detected = db.Column(db.Boolean, default=False)    # Flag for weapon detection
    camera = db.relationship('Camera', backref=db.backref('incidents', lazy=True))

class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    email = db.Column(db.String(100))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# ------------------------
# Helper Functions
# ------------------------
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points 
    on the Earth using the haversine formula.
    """
    R = 6371  # Earth radius in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# ------------------------
# Background Task for Camera Status Check
# ------------------------
def check_camera_status():
    """Checks all cameras every 15 minutes and updates their working status."""
    while True:
        with app.app_context():
            cameras = Camera.query.all()
            for camera in cameras:
                cap = cv2.VideoCapture(camera.url)
                if cap.isOpened():
                    camera.working = True
                    cap.release()
                else:
                    camera.working = False
                db.session.commit()
        time.sleep(900)  # Sleep for 15 minutes (900 seconds)

# Start the background thread when the app launches
camera_status_thread = threading.Thread(target=check_camera_status, daemon=True)
camera_status_thread.start()

# ------------------------
# Recording Functionality
# ------------------------
# Global dictionary to store active recording threads
recording_threads = {}

import threading
import time
import cv2
import numpy as np
import queue
from collections import deque
from keras.models import load_model
from ultralytics import YOLO

# Note: 'app', 'Incident', 'db', and 'recording_threads' are assumed to be defined in your main app context.

class RecordingThread(threading.Thread):
    def __init__(self, key, camera_url, output_file, duration=7200):
        super(RecordingThread, self).__init__()
        self.key = key  # key is a tuple (incident_id, camera_id)
        self.camera_url = camera_url
        self.output_file = output_file
        self.duration = duration  # in seconds (e.g., 2 hours)
        self.stopped = False
        self.violence_flag = False  # Flag for violence detection
        self.weapon_flag = False    # Flag for weapon detection

    def run(self):
        cap = cv2.VideoCapture(self.camera_url)
        if not cap.isOpened():
            print(f"Failed to open video stream from {self.camera_url}")
            if self.key in recording_threads:
                del recording_threads[self.key]
            return

        # Create a thread-safe queue to buffer captured frames.
        frame_queue = queue.Queue(maxsize=128)
        
        # Producer: Continuously capture frames and add them to the queue.
        def frame_reader():
            while not self.stopped:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                # If the queue is full, discard the oldest frame.
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(frame)
            cap.release()
        
        # Start the frame capture thread.
        reader_thread = threading.Thread(target=frame_reader, daemon=True)
        reader_thread.start()
        
        # Load the violence detection model.
        try:
            violence_model = load_model('modelnew.h5')
        except Exception as e:
            print(f"Failed to load violence detection model: {e}")
            violence_model = None

        # Load the YOLO weapon detection model.
        try:
            yolo_model = YOLO('best.pt')
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            yolo_model = None

        # Wait for a frame to determine video dimensions.
        try:
            sample_frame = frame_queue.get(timeout=5)
        except queue.Empty:
            print("No frames received from camera.")
            if self.key in recording_threads:
                del recording_threads[self.key]
            return

        height, width = sample_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # Desired output FPS. Adjust as needed.
        desired_fps = 32.0
        out = cv2.VideoWriter(self.output_file, fourcc, desired_fps, (width, height))
        frame_interval = 1 / desired_fps

        # Deques for averaging predictions.
        violence_Q = deque(maxlen=128)
        weapon_Q = deque(maxlen=128)

        start_time = time.time()
        last_frame_time = time.time()
        # Continue processing until duration expires OR the queue is empty (after stop is called).
        while (time.time() - start_time) < self.duration or not frame_queue.empty():
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                if self.stopped:
                    break
                continue

            # Enforce constant frame interval.
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()

            output_frame = frame.copy()

            # -------- Violence Detection --------
            if violence_model is not None:
                try:
                    proc_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    proc_frame = cv2.resize(proc_frame, (128, 128)).astype("float32") / 255.0
                    pred = violence_model.predict(np.expand_dims(proc_frame, axis=0))[0][0]
                    violence_Q.append(pred)
                    avg_pred = np.mean(violence_Q)
                    if avg_pred > 0.50 and not self.violence_flag:
                        self.violence_flag = True
                        with app.app_context():
                            incident = Incident.query.get(self.key[0])
                            if incident:
                                incident.violence_detected = True
                                db.session.commit()
                except Exception as e:
                    print(f"Error during violence detection: {e}")

            violence_text = "Violence Detected" if self.violence_flag else "No Violence"
            violence_color = (0, 0, 255) if self.violence_flag else (0, 255, 0)
            cv2.putText(output_frame, violence_text, (35, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.25, violence_color, 3)

            # -------- Weapon Detection --------
            if yolo_model is not None:
                try:
                    results = yolo_model(frame)
                    weapon_detected_in_frame = False
                    for result in results:
                        classes = result.names
                        for pos, detection in enumerate(result.boxes.xyxy):
                            conf_val = float(result.boxes.conf[pos])
                            if conf_val >= 0.5:
                                weapon_detected_in_frame = True
                                xmin, ymin, xmax, ymax = map(int, detection.tolist())
                                label = f"{classes[int(result.boxes.cls[pos])]} {conf_val:.2f}"
                                color = (0, int(result.boxes.cls[pos]) % 256, 255)
                                cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), color, 2)
                                cv2.putText(output_frame, label, (xmin, max(ymin - 10, 0)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    if weapon_detected_in_frame:
                        weapon_Q.append(1)
                    else:
                        weapon_Q.append(0)
                    if np.mean(weapon_Q) > 0.5 and not self.weapon_flag:
                        self.weapon_flag = True
                        with app.app_context():
                            incident = Incident.query.get(self.key[0])
                            if incident:
                                incident.weapon_detected = True
                                db.session.commit()
                except Exception as e:
                    print(f"Error during weapon detection: {e}")

            weapon_text = "Weapon Detected" if self.weapon_flag else "No Weapon"
            weapon_color = (0, 0, 255) if self.weapon_flag else (0, 255, 0)
            cv2.putText(output_frame, weapon_text, (35, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.25, weapon_color, 3)

            # Write the processed frame to the output video.
            out.write(output_frame)

        out.release()
        print(f"Recording saved to {self.output_file}")
        if self.key in recording_threads:
            del recording_threads[self.key]

    def stop(self):
        self.stopped = True

        
# ------------------------
# Routes for Public Pages
# ------------------------
@app.route('/')
def index():
    return render_template('index.html')

# ------------------------
# User Registration & Login
# ------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        admin_code = request.form.get('admin_code')  # New field in registration form

        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        hashed = generate_password_hash(password)
        role = 'admin' if admin_code == "71124" else 'user'
        new_user = User(username=username, password_hash=hashed, role=role)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful, please login', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Logged in successfully', 'success')
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('user_portal'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

# ------------------------
# Admin Routes
# ------------------------
@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash("Unauthorized access", "danger")
        return redirect(url_for('user_portal'))
    cameras = Camera.query.all()
    incidents = Incident.query.order_by(Incident.timestamp.desc()).all()
    active_recordings = []
    for (incident_id, camera_id), thread in recording_threads.items():
        camera = Camera.query.get(camera_id)
        active_recordings.append({
            'incident_id': incident_id,
            'camera_id': camera_id,
            'camera_url': camera.url if camera else 'Unknown'
        })
    return render_template('admin_dashboard.html', cameras=cameras, incidents=incidents, active_recordings=active_recordings)

@app.route('/admin/add_camera', methods=['GET', 'POST'])
@login_required
def add_camera():
    if current_user.role != 'admin':
        flash("Unauthorized access", "danger")
        return redirect(url_for('login'))
    if request.method == 'POST':
        url_camera = request.form.get('url')
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except ValueError:
            flash('Invalid latitude or longitude', 'danger')
            return redirect(url_for('add_camera'))
        new_camera = Camera(url=url_camera, latitude=latitude, longitude=longitude)
        db.session.add(new_camera)
        db.session.commit()
        flash('Camera added successfully!', 'success')
        return redirect(url_for('admin_dashboard'))
    return render_template('add_camera.html')

@app.route('/admin/stop_recording', methods=['POST'])
@login_required
def stop_recording():
    if current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    incident_id = request.form.get('incident_id')
    camera_id = request.form.get('camera_id')
    try:
        incident_id = int(incident_id)
        camera_id = int(camera_id)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid parameters'}), 400
    key = (incident_id, camera_id)
    if key in recording_threads:
        recording_threads[key].stop()  # Stop the thread
        del recording_threads[key]
        return jsonify({'message': 'Recording stopped.'})
    else:
        return jsonify({'error': 'Recording not found.'}), 404

# ------------------------
# User Routes
# ------------------------
@app.route('/user')
@login_required
def user_portal():
    if current_user.role != 'user':
        flash("Unauthorized access", "danger")
        return redirect(url_for('login'))
    contacts = Contact.query.filter_by(user_id=current_user.id).all()
    return render_template('user_portal.html', contacts=contacts)

@app.route('/user/sos', methods=['POST'])
@login_required
def sos():
    if current_user.role != 'user':
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    user_lat = data.get('latitude')
    user_lng = data.get('longitude')
    if user_lat is None or user_lng is None:
        return jsonify({'error': 'Invalid data'}), 400

    try:
        user_lat = float(user_lat)
        user_lng = float(user_lng)
    except ValueError:
        return jsonify({'error': 'Invalid latitude or longitude'}), 400

    threshold = 50.0  # km threshold for nearby cameras
    cameras = Camera.query.all()
    nearby_cameras = []

    for camera in cameras:
        # Only use cameras that are working
        if not camera.working:
            continue

        distance = haversine(user_lat, user_lng, camera.latitude, camera.longitude)
        if distance <= threshold:
            incident = Incident(user_latitude=user_lat, user_longitude=user_lng, camera_id=camera.id)
            db.session.add(incident)
            db.session.commit()

            os.makedirs("recordings", exist_ok=True)
            output_file = f"recordings/recording_incident_{incident.id}_camera_{camera.id}.avi"
            key = (incident.id, camera.id)
            rec_thread = RecordingThread(key=key, camera_url=camera.url, output_file=output_file, duration=7200)
            rec_thread.start()
            recording_threads[key] = rec_thread

            nearby_cameras.append({
                'id': camera.id,
                'url': camera.url,
                'distance': round(distance, 2),
                'incident_id': incident.id
            })

    # Send notifications to emergency contacts
    contacts = Contact.query.filter_by(user_id=current_user.id).all()
    for contact in contacts:
        if contact.email:
            try:
                msg = Message("Emergency Alert",
                              recipients=[contact.email])
                msg.body = (f"Dear {contact.name},\n\n"
                            f"An emergency alert has been triggered by {current_user.username}.\n"
                            f"Location: Latitude {user_lat}, Longitude {user_lng}\n\n"
                            "Please take immediate action.\n\n"
                            "This is an automated message.")
                mail.send(msg)
            except Exception as e:
                print(f"Failed to send email to {contact.email}: {str(e)}")
    response = {
        'message': 'SOS received. Incident(s) recorded and recording started for nearby cameras.',
        'nearby_cameras': nearby_cameras
    }
    return jsonify(response), 200

@app.route('/user/add_contact', methods=['POST'])
@login_required
def add_contact():
    if current_user.role != 'user':
        flash('Only users can add emergency contacts', 'danger')
        return redirect(url_for('user_portal'))
    name = request.form.get('name')
    phone = request.form.get('phone')
    email = request.form.get('email')
    if not name or not phone:
        flash('Name and phone are required', 'danger')
        return redirect(url_for('user_portal'))
    new_contact = Contact(name=name, phone=phone, email=email, user_id=current_user.id)
    db.session.add(new_contact)
    db.session.commit()
    flash('Emergency contact added successfully!', 'success')
    return redirect(url_for('user_portal'))

# ------------------------
# Run the App
# ------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
