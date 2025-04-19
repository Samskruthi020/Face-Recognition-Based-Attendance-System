import cv2
import os
from flask import Flask, request, render_template, send_file
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Defining Flask App
app = Flask(__name__)

# Increased number of images for better training
nimgs = 30  # Increased from 10 to 30 images per user

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Improved face detection parameters
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_detector_params = {
    'scaleFactor': 1.1,
    'minNeighbors': 5,
    'minSize': (30, 30),
    'flags': cv2.CASCADE_SCALE_IMAGE
}

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

# Function to migrate old CSV to new format
def migrate_attendance_csv():
    csv_path = f'Attendance/Attendance-{datetoday}.csv'
    if os.path.exists(csv_path):
        try:
            # Try to read the existing CSV with error handling
            try:
                df = pd.read_csv(csv_path)
            except pd.errors.ParserError:
                # If there's a parsing error, read the file manually
                with open(csv_path, 'r') as f:
                    lines = f.readlines()
                # Create a new DataFrame with the correct format
                data = []
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) == 3:  # Old format
                        data.append([parts[0], parts[1], 'CSE', parts[2]])  # Add default branch
                    elif len(parts) == 4:  # New format
                        data.append(parts)
                df = pd.DataFrame(data, columns=['Name', 'Roll', 'Branch', 'Time'])
            
            # Ensure all required columns exist
            if 'Branch' not in df.columns:
                df['Branch'] = 'CSE'  # Add Branch column with default value
            
            # Save with new format
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error migrating CSV: {str(e)}")
            # If all else fails, create a new file with the correct format
            with open(csv_path, 'w') as f:
                f.write('Name,Roll,Branch,Time')
    else:
        # Create new file with new format
        with open(csv_path, 'w') as f:
            f.write('Name,Roll,Branch,Time')

# Migrate existing CSV file to new format
migrate_attendance_csv()

# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# extract the face from an image with improved parameters
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, **face_detector_params)
        return face_points
    except:
        return []

# Data augmentation function
def augment_image(img):
    augmented_images = []
    
    # Original image
    augmented_images.append(img)
    
    # Flip horizontally
    augmented_images.append(cv2.flip(img, 1))
    
    # Adjust brightness
    brightness = np.random.uniform(0.7, 1.3)
    augmented_images.append(cv2.convertScaleAbs(img, alpha=brightness, beta=0))
    
    # Add slight blur
    augmented_images.append(cv2.GaussianBlur(img, (3,3), 0))
    
    return augmented_images

# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            if img is not None:
                # Apply data augmentation
                augmented_images = augment_image(img)
                for aug_img in augmented_images:
                    # Resize face to 50x50 to match model's expected 7500 features (50*50*3)
                    resized_face = cv2.resize(aug_img, (50, 50))
                    faces.append(resized_face.ravel())
                    labels.append(user)
    
    if len(faces) > 0:
        faces = np.array(faces)
        # Create a pipeline with standardization and KNN
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=3, weights='distance', metric='cosine'))
        ])
        pipeline.fit(faces, labels)
        joblib.dump(pipeline, 'static/face_recognition_model.pkl')
        return True
    return False

# Extract info from today's attendance file in attendance folder
def extract_attendance():
    try:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        if 'Branch' not in df.columns:
            df['Branch'] = 'CSE'  # Add Branch column if it doesn't exist
        names = df['Name']
        rolls = df['Roll']
        times = df['Time']
        branches = df['Branch']
        l = len(df)
        return names, rolls, times, branches, l
    except Exception as e:
        print(f"Error reading attendance file: {str(e)}")
        # Return empty data if there's an error
        return [], [], [], [], 0

# Add Attendance of a specific user
def add_attendance(name):
    if name == "Unknown":
        return
        
    try:
        username = name.split('_')[0]
        userid = name.split('_')[1]
        branch = "CSE"  # Default branch
        current_time = datetime.now().strftime("%H:%M:%S")

        # Ensure file exists and has correct format
        migrate_attendance_csv()
        
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        if int(userid) in list(df['Roll']):
            df.loc[df['Roll'] == int(userid), 'Time'] = current_time
            df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)
        else:
            with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                f.write(f'\n{username},{userid},{branch},{current_time}')
    except Exception as e:
        print(f"Error adding attendance: {str(e)}")

## A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

## A function to delete a user folder 
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)

# Identify face using ML model
def identify_face(facearray):
    try:
        model = joblib.load('static/face_recognition_model.pkl')
        return model.predict(facearray)
    except Exception as e:
        print(f"Error in face identification: {str(e)}")
        return ["Unknown"]

################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    names, rolls, times, branches, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, branches=branches, l=l, totalreg=totalreg(), datetoday2=datetoday2)

## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)

    ## if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/')==[]:
        os.remove('static/face_recognition_model.pkl')
    
    try:
        train_model()
    except:
        pass

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, branches, l = extract_attendance()
    marked_attendance = set()  # Keep track of already marked attendance

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, branches=branches, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            # Resize face to 50x50 to match model's expected 7500 features (50*50*3)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            
            # Update attendance only for the person currently in camera
            if identified_person != "Unknown":
                add_attendance(identified_person)
                marked_attendance.add(identified_person)
                # Update the attendance list after marking
                names, rolls, times, branches, l = extract_attendance()
            
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show attendance marked message
            if identified_person in marked_attendance:
                cv2.putText(frame, 'Attendance Marked', (x+5, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, branches, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, branches=branches, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# A function to add a new user with improved face capture
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    branch = request.form.get('branch', 'CSE')  # Get branch, default to CSE
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while i < nimgs:
        _, frame = cap.read()
        faces = extract_faces(frame)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            
            # Capture image every 5 frames to ensure variety
            if j % 5 == 0:
                face_img = frame[y:y+h, x:x+w]
                # Resize face to 50x50 to match model's expected 7500 features (50*50*3)
                face_img = cv2.resize(face_img, (50, 50))
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, face_img)
                i += 1
            j += 1
            
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if i > 0:  # Only train if we captured some images
        print('Training Model')
        if train_model():
            print('Model trained successfully')
        else:
            print('Failed to train model - no valid images')
    
    names, rolls, times, branches, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, branches=branches, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/download')
def download():
    path = f'Attendance/Attendance-{datetoday}.csv'
    return send_file(path, as_attachment=True)

# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
