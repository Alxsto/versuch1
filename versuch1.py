import streamlit as st
import os
import json
from datetime import datetime
from PIL import Image
import numpy as np
import tensorflow as tf

# Ordnerstruktur
DATA_FILE = "data.json"
IMAGE_FOLDER = "images"
MODEL_PATH = "model/keras_model.h5"
LABELS_PATH = "model/labels.txt"

os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Daten laden
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

# Daten speichern
def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Modell laden
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

# Bild klassifizieren
def classify_image(image, model, labels):
    img = image.resize((224, 224))
    img_array = np.asarray(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    return labels[index], float(confidence)

# UI
st.title("🔍 Digitales Fundbüro")

tab1, tab2 = st.tabs(["📤 Gegenstand melden", "🔎 Suche"])

data = load_data()

# =============================
# TAB 1: Upload
# =============================
with tab1:
    st.header("Gefundenen Gegenstand hochladen")

    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])
    description = st.text_input("Beschreibung")
    location = st.text_input("Fundort")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Vorschau")

    if st.button("Speichern"):
        if uploaded_file:
            # Bild speichern
            filename = f"{datetime.now().timestamp()}.png"
            filepath = os.path.join(IMAGE_FOLDER, filename)

            image.save(filepath)

            # KI Klassifikation
            try:
                model, labels = load_model()
                label, confidence = classify_image(image, model, labels)
            except Exception as e:
                label, confidence = "Unbekannt", 0

            entry = {
                "description": description,
                "location": location,
                "image": filepath,
                "date": str(datetime.now()),
                "ai_label": label,
                "confidence": confidence
            }

            data.append(entry)
            save_data(data)

            st.success(f"Gespeichert! KI erkennt: {label} ({confidence:.2f})")
        else:
            st.error("Bitte ein Bild hochladen")

# =============================
# TAB 2: Suche
# =============================
with tab2:
    st.header("Fundstücke durchsuchen")

    search_text = st.text_input("Suche (Beschreibung oder KI-Erkennung)")
    
    filtered_data = data

    if search_text:
        filtered_data = [
            item for item in data
            if search_text.lower() in item["description"].lower()
            or search_text.lower() in item["ai_label"].lower()
        ]

    for item in reversed(filtered_data):
        st.subheader(item["description"])
        st.write(f"📍 Ort: {item['location']}")
        st.write(f"🤖 KI: {item['ai_label']} ({item['confidence']:.2f})")
        st.write(f"🕒 Datum: {item['date']}")
        st.image(item["image"], width=200)
        st.markdown("---")
