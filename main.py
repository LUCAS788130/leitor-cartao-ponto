import streamlit as st
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import numpy as np
import re
from datetime import datetime

st.set_page_config(layout="wide")
st.title("Leitor Inteligente de Cartão de Ponto")

uploaded_file = st.file_uploader("Envie o cartão de ponto (PDF, imagem, foto digitalizada)")

def preprocess_image(pil_image):
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    return Image.fromarray(gray)

def ocr_image(image):
    return pytesseract.image_to_string(image, lang="por")

def extract_text(file):
    text = ""
    if file.type == "application/pdf":
        pages = convert_from_bytes(file.read())
        for page in pages:
            processed = preprocess_image(page)
            text += ocr_image(processed)
    else:
        image = Image.open(file)
        processed = preprocess_image(image)
        text += ocr_image(processed)
    return text

def parse_times(text):
    # Datas no formato 01/03/2026
    dates = re.findall(r'\d{2}/\d{2}/\d{4}', text)

    # Horários no formato 08:00, 17:59 etc
    times = re.findall(r'\d{2}:\d{2}', text)

    registros = []
    t = 0

    for d in dates:
        entrada1 = times[t] if t < len(times) else ""
        saida1   = times[t+1] if t+1 < len(times) else ""
        entrada2 = times[t+2] if t+2 < len(times) else ""
        saida2   = times[t+3] if t+3 < len(times) else ""
        t += 4

        registros.append({
            "Data": d,
            "Entrada1": entrada1,
            "Saida1": saida1,
            "Entrada2": entrada2,
            "Saida2": saida2
        })

    return pd.DataFrame(registros)

if uploaded_file:
    st.info("Lendo cartão de ponto...")

    raw_text = extract_text(uploaded_file)

    st.subheader("Texto reconhecido pelo OCR")
    st.text(raw_text)

    df = parse_times(raw_text)

    st.subheader("Modelo final gerado")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar CSV no modelo", csv, "cartao_ponto.csv", "text/csv")
