# Face Recognition Attendance System

A Face Recognition Attendance System built using Python, OpenCV, face_recognition, and Streamlit. The application recognizes registered faces and marks attendance automatically.

## Features

- Face detection and recognition
- Automatic attendance marking
- User-friendly Streamlit interface
- Train custom face dataset
- CSV attendance records

## Technologies Used

- Python
- OpenCV
- face_recognition
- Streamlit
- NumPy
- Pandas

## Project Structure

```
Face-Recognition-Model/
│── app.py
│── train.py
│── requirements.txt
│── README.md
│── dataset/
│── attendance.csv
│── face_data.pkl
```

## Installation

1. Clone the repository

```bash
git clone https://github.com/your-username/Face-Recognition-Model.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Train the model

```bash
python train.py
```

4. Run the application

```bash
streamlit run app.py
```

## Dataset

Create folders inside the `dataset` directory. Each folder should be named after a person and contain their face images.

Example:

```
dataset/
├── Person1/
├── Person2/
├── Person3/
```

## Future Improvements

- Real-time webcam recognition
- Database integration
- Email notifications
- Face mask detection

## Author

Yash Prakash Sharma

B.Tech CSE (2027)
