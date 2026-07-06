
import streamlit as st
import face_recognition
import numpy as np
import cv2
import pickle
from PIL import Image
import pandas as pd
from datetime import datetime

# ======================================
# PAGE CONFIGURATION
# ======================================
recognized_people = []

st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🧑",
    layout="wide"
)

# ======================================
# CUSTOM CSS
# ======================================

st.markdown("""
<style>

.main{
    background-color:#f5f5f5;
}

h1{
    color:#0066cc;
    text-align:center;
}

.stButton>button{
    width:100%;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ======================================
# TITLE
# ======================================

st.title("🧑 Face Recognition System")

st.write(
    "Upload an image to recognize one or more registered faces."
)

# ======================================
# LOAD TRAINED MODEL
# ======================================

try:

    with open("face_data.pkl", "rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names = data["names"]

except FileNotFoundError:

    st.error("face_data.pkl not found.")
    st.info("Run train.py first.")

    st.stop()

except Exception as e:

    st.error(e)

    st.stop()

# ======================================
# SIDEBAR
# ======================================

st.sidebar.title("📊 Model Information")

st.sidebar.success(
    f"Registered Persons : {len(set(known_names))}"
)

st.sidebar.success(
    f"Training Images : {len(known_names)}"
)

tolerance = st.sidebar.slider(
    "Recognition Sensitivity",
    0.30,
    0.80,
    0.50,
    0.05
)

st.sidebar.markdown("---")

st.sidebar.write("Developed using")

st.sidebar.write("✔ Streamlit")

st.sidebar.write("✔ OpenCV")

st.sidebar.write("✔ face_recognition")

st.sidebar.write("✔ NumPy")

# ======================================
# IMAGE UPLOAD
# ======================================

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)
# ======================================
# FACE RECOGNITION
# ======================================



recognized_people = []

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    key="upload_image"
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # ALL your face code must be inside here
    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)

    st.markdown("---") 
    
    success, buffer = cv2.imencode(".jpg",cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    )
    
    for (top, right, bottom, left), face_encoding in zip(
            face_locations,
            face_encodings
    ):

            matches = face_recognition.compare_faces(
                known_encodings,
                face_encoding,
                tolerance=tolerance
            )

            face_distances = face_recognition.face_distance(
                known_encodings,
                face_encoding
            ) 

            name = "Unknown"
            confidence = 0

            if len(face_distances) > 0:

                best_match = np.argmin(face_distances)

                if matches[best_match]:

                    name = known_names[best_match]

                    confidence = (1 - face_distances[best_match]) * 100

            recognized_people.append(
                {
                    "Name": name,
                    "Confidence": round(confidence,2)
                }
            )

            # Green Rectangle
            cv2.rectangle(
                image_np,
                (left, top),
                (right, bottom),
                (0,255,0),
                3
            )

            # Label Background
            cv2.rectangle(
                image_np,
                (left, bottom-35),
                (right, bottom),
                (0,255,0),
                cv2.FILLED
            )

            # Label Text
            label = f"{name} ({confidence:.1f}%)"

            cv2.putText(
                image_np,
                label,
                (left+6,bottom-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )
            st.markdown("---")
            
            st.subheader("Recognized Image")
            st.image(
            image_np,
            channels="RGB",
            use_container_width=True
        )
            st.markdown("---")
            st.subheader("Recognition Results")
            df = pd.DataFrame(recognized_people)
            st.dataframe(
            df,
            use_container_width=True
        )
        # ======================================
# ATTENDANCE SYSTEM
# ======================================

attendance = []

for person in recognized_people:

    if person["Name"] != "Unknown":

        attendance.append(
            {
                "Name": person["Name"],
                "Confidence": person["Confidence"],
                "Date": datetime.now().strftime("%d-%m-%Y"),
                "Time": datetime.now().strftime("%H:%M:%S")
            }
        )

if len(attendance) > 0:

    attendance_df = pd.DataFrame(attendance)

    st.markdown("---")

    st.subheader("Attendance")

    st.dataframe(attendance_df, use_container_width=True)

    attendance_df.to_csv(
        "attendance.csv",
        index=False
    )

    with open("attendance.csv", "rb") as f:

        st.download_button(
            label="📥 Download Attendance CSV",
            data=f,
            file_name="attendance.csv",
            mime="text/csv"
        )

# ======================================
# DOWNLOAD IMAGE
# ======================================

if uploaded_file is not None:

    st.markdown("---")

    success, buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    )

    if success:
        st.download_button(
            label="📥 Download Recognized Image",
            data=buffer.tobytes(),
            file_name="recognized_image.jpg",
            mime="image/jpeg"
        )
# ======================================
# MODEL INFORMATION
# ======================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Registered Persons",
    len(set(known_names))
)

col2.metric(
    "Training Images",
    len(known_names)
)

if uploaded_file is not None:
    faces_detected = len(face_locations)
else:
    faces_detected = 0

col3.metric(
    "Faces Detected",
    faces_detected
)

# ======================================
# FOOTER
# ======================================

st.markdown("---")

st.markdown(
    """
    <center>

    ### Face Recognition System

    Developed using

    Streamlit • OpenCV • face_recognition • NumPy

    </center>
    """,
    unsafe_allow_html=True
)
