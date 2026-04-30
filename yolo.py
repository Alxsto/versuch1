import streamlit as st
import os
import sqlite3
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

# --- Setup ---
st.set_page_config(page_title="Fundbüro", layout="wide")
st.title("🔎 Digitales Fundbüro")

UPLOAD_FOLDER = "uploads"
DB_FILE = "fundbuero.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Datenbank ---
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    label TEXT,
    date TEXT
)
""")
conn.commit()

# --- YOLO Modell sicher laden ---
@st.cache_resource
def load_model():
    model_path = "yolov8n.pt"

    if not os.path.exists(model_path):
        st.error("YOLO Modell fehlt! Bitte 'yolov8n.pt' hochladen.")
        return None

    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error("Fehler beim Laden des Modells")
        st.text(str(e))
        return None

model = load_model()

# Wenn Modell nicht geladen → stoppen
if model is None:
    st.stop()

# --- Sidebar ---
menu = st.sidebar.selectbox("Menü", ["Upload", "Suche"])

# --- Upload ---
if menu == "Upload":
    st.header("📤 Fundstück hochladen")

    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(filepath, caption="Bild", use_column_width=True)

        # --- YOLO Analyse ---
        with st.spinner("Erkenne Objekt..."):
            try:
                results = model(filepath)

                labels = []
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = model.names[cls]
                        labels.append(label)

                detected_label = ", ".join(set(labels)) if labels else "Unbekannt"

            except Exception as e:
                detected_label = "Fehler bei Erkennung"
                st.error(str(e))

        st.success(f"Erkannt: {detected_label}")

        # --- Speichern ---
        c.execute(
            "INSERT INTO items (filename, label, date) VALUES (?, ?, ?)",
            (uploaded_file.name, detected_label, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()

        st.success("Gespeichert!")

# --- Suche ---
elif menu == "Suche":
    st.header("🔍 Fundstücke durchsuchen")

    search = st.text_input("Suche (z.B. Handy, Tasche)")

    if search:
        c.execute("SELECT * FROM items WHERE label LIKE ?", ('%' + search + '%',))
    else:
        c.execute("SELECT * FROM items ORDER BY date DESC")

    results = c.fetchall()

    for item in results:
        filepath = os.path.join(UPLOAD_FOLDER, item[1])

        if os.path.exists(filepath):
            st.image(filepath, width=200)

        st.write(f"**Erkannt:** {item[2]}")
        st.write(f"**Datum:** {item[3]}")
        st.markdown("---")
