# Face Recognition Based Attendance System

A simple and efficient attendance system that uses face recognition to mark attendance automatically.

## Features
- Automatic face detection and recognition
- Real-time attendance marking
- Easy student registration
- Daily attendance reports
- Download attendance in CSV format

## Installation Steps

1. **Install Python**
   - Download and install Python 3.8 or higher from [python.org](https://www.python.org/downloads/)

2. **Install Required Packages**
   Open Command Prompt (CMD) and run these commands:
   ```bash
   pip install flask
   pip install opencv-python
   pip install numpy
   pip install pandas
   pip install scikit-learn
   pip install joblib
   ```

3. **Download the Project**
   - Download this project folder to your computer
   - Open the folder in your preferred code editor

4. **Run the Application**
   - Open Command Prompt (CMD)
   - Navigate to the project folder
   - Run the command:
   ```bash
   python app.py
   ```
   - Open your web browser and go to: `http://localhost:5000`

## How to Use

1. **Add New Students**
   - Click on "Add New User"
   - Enter student name and roll number
   - Click "Take Images" to capture student's face
   - The system will capture 30 images for better recognition

2. **Mark Attendance**
   - Click "Take Attendance"
   - The system will automatically detect and recognize faces
   - Attendance will be marked automatically

3. **View Attendance**
   - The home page shows today's attendance
   - You can download the attendance report in CSV format

4. **Manage Students**
   - View all registered students
   - Delete students if needed

## Requirements
- Webcam
- Good lighting conditions
- Python 3.8 or higher
- Internet connection (for first-time package installation)

## Troubleshooting
- Make sure your webcam is working
- Ensure good lighting for better face detection
- If face detection fails, try adjusting your position or lighting
- Restart the application if you encounter any errors

## Note
- The system works best with clear face images
- Keep a distance of 1-2 feet from the camera
- Make sure your face is well-lit and clearly visible
