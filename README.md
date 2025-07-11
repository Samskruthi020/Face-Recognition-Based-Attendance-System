# Face Recognition Based Attendance System

A simple and efficient attendance system that uses face recognition to mark attendance automatically. Now includes a **mobile app** for on-the-go attendance management!

## Features
- Automatic face detection and recognition
- Real-time attendance marking
- Easy student registration
- Daily attendance reports
- Download attendance in CSV format
- **📱 Mobile Progressive Web App (PWA)**
- **📸 Mobile camera integration for face capture**
- **🔄 Real-time attendance via mobile devices**
- **📊 Mobile-optimized dashboard and UI**

## Mobile App Features

### 📱 Progressive Web App
- **Installable on mobile devices** like a native app
- **Offline support** with service worker
- **Mobile-optimized interface** with touch-friendly controls
- **Camera integration** for real-time face recognition
- **Real-time attendance marking** from your phone

### How to Access Mobile App
1. Run the application as described below
2. Open your mobile browser and go to: `http://your-server:5000/mobile`
3. For best experience, **install the app**:
   - **iOS**: Tap Share → "Add to Home Screen"
   - **Android**: Tap Menu → "Add to Home screen"
   - **Desktop**: Look for install prompt in browser

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
   pip install pillow
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

4. **Mobile App Usage**
   - Access `/mobile` for the mobile-optimized interface
   - Install as PWA for native app experience
   - Use mobile camera for real-time attendance marking
   - Register new users directly from mobile device

5. **Manage Students**
   - View all registered students
   - Delete students if needed

## Requirements
- Webcam or mobile device with camera
- Good lighting conditions
- Python 3.8 or higher
- Internet connection (for first-time package installation)
- **For mobile**: Modern mobile browser (Chrome, Safari, Firefox)

## API Endpoints (for mobile integration)

The system provides REST API endpoints for mobile app integration:

- `GET /api/attendance` - Get today's attendance data
- `GET /api/users` - Get all registered users  
- `POST /api/add_user` - Register new user via image upload
- `POST /api/recognize` - Recognize face and mark attendance
- `DELETE /api/delete_user/<user>` - Delete a user

See `MOBILE_README.md` for detailed API documentation.

## Troubleshooting
- Make sure your webcam is working
- Ensure good lighting for better face detection
- If face detection fails, try adjusting your position or lighting
- Restart the application if you encounter any errors

## Note
- The system works best with clear face images
- Keep a distance of 1-2 feet from the camera
- Make sure your face is well-lit and clearly visible
