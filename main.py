import streamlit as st
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import numpy as np
import re

st.set_page_config(layout="wide")
st.title("Leitor Inteligente de Cartão de Ponto")

uploaded_file = st.file_uploader("Envie o cartão de ponto (PDF, imagem, foto)")

# ---------- OCR ----------

def preprocess(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    return Image.fromarray(gray)

def ocr(img):
    try:
        return pytesseract.image_to_string(img, lang="por")
    except:
        return pytesseract.image_to_string(img)

def extract_text(file):
    text = ""
    if file.type == "application/pdf":
        pages = convert_from_bytes(file.read())
        for p in pages:
            text += ocr(preprocess(p))
    else:
        img = Image.open(file)
        text += ocr(preprocess(img))
    return text

# ---------- PARSER INTELIGENTE ----------

def organizar_por_dia(texto):
    linhas = texto.split("\n")
    padrao_data = r"\d{2}/\d{2}/\d{4}"
    padrao_hora = r"\d{2}:\d{2}"

    dados = []
    data_atual = None
    horas_do_dia = []

    for linha in linhas:
        data = re.search(padrao_data, linha)
        horas = re.findall(padrao_hora, linha)

        if data:
            # salva o dia anterior
            if data_atual:
                dados.append([data_atual] + horas_do_dia[:4])
            data_atual = data.group()
            horas_do_dia = horas
        else:
            horas_do_dia.extend(horas)

    # salva último dia
    if data_atual:
        dados.append([data_atual] + horas_do_dia[:4])

    df = pd.DataFrame(dados, columns=[
        "Data", "Entrada1", "Saida1", "Entrada2", "Saida2"
    ])

    return df

# ---------- APP ----------

if uploaded_file:
    st.info("Lendo cartão...")
    texto = extract_text(uploaded_file)

    st.subheader("Texto identificado (OCR)")
    st.text(texto)

    df = organizar_por_dia(texto)

    st.subheader("Ajuste manual antes de exportar")
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    csv = edited_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar CSV no modelo correto",
        csv,
        "cartao_ponto_ajustado.csv",
        "text/csv"
    )
