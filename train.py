
import os
import pickle
import face_recognition
from tqdm import tqdm

# ==============================
# Configuration
# ==============================

DATASET_DIR = "dataset"
MODEL_FILE = "face_data.pkl"

known_encodings = []
known_names = []

print("=" * 60)
print(" FACE RECOGNITION MODEL TRAINING ")
print("=" * 60)

# ==============================
# Check Dataset
# ==============================

if not os.path.exists(DATASET_DIR):
    print(f"\nDataset folder '{DATASET_DIR}' not found.")
    exit()

persons = [
    p for p in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, p))
]

if len(persons) == 0:
    print("No person folders found inside dataset.")
    exit()

print(f"\nFound {len(persons)} persons.\n")

# ==============================
# Training
# ==============================

for person in persons:

    person_path = os.path.join(DATASET_DIR, person)

    images = os.listdir(person_path)

    print(f"\nTraining : {person}")

    for image_name in tqdm(images):

        image_path = os.path.join(person_path, image_name)

        try:

            image = face_recognition.load_image_file(image_path)

            locations = face_recognition.face_locations(image)

            encodings = face_recognition.face_encodings(image, locations)

            if len(encodings) == 0:
                print(f"\nNo face detected in {image_name}")
                continue

            known_encodings.append(encodings[0])
            known_names.append(person)

        except Exception as e:
            print(f"\nError : {image_name}")
            print(e)

# ==============================
# Save Model
# ==============================

data = {
    "encodings": known_encodings,
    "names": known_names
}

with open(MODEL_FILE, "wb") as f:
    pickle.dump(data, f)

# ==============================
# Summary
# ==============================

print("\n")
print("=" * 60)
print(" TRAINING COMPLETED ")
print("=" * 60)

print(f"Total Images      : {len(known_names)}")
print(f"Unique Persons    : {len(set(known_names))}")
print(f"Model Saved       : {MODEL_FILE}")

print("=" * 60)
