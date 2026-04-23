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

uploaded_file = st.file_uploader("Envie o cartão de ponto")

def preparar_imagem(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,15,4)
    return thresh

def detectar_tabela(thresh):
    horizontal = thresh.copy()
    vertical = thresh.copy()

    cols = horizontal.shape[1]
    horizontal_size = cols // 20
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size,1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    rows = vertical.shape[0]
    vertical_size = rows // 20
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,vertical_size))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    mask = horizontal + vertical
    return mask

def extrair_celulas(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    celulas = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 80 and h > 25:
            celulas.append((x, y, w, h))

    celulas = sorted(celulas, key=lambda b: (b[1], b[0]))
    return celulas

def ocr_celula(img, box):
    x, y, w, h = box
    crop = img[y:y+h, x:x+w]
    pil = Image.fromarray(crop)
    try:
        return pytesseract.image_to_string(pil, lang="por").strip()
    except:
        return pytesseract.image_to_string(pil).strip()

def processar_pagina(pil_img):
    thresh = preparar_imagem(pil_img)
    mask = detectar_tabela(thresh)
    celulas = extrair_celulas(thresh, mask)

    dados = []
    linha_atual = []

    for box in celulas:
        texto = ocr_celula(thresh, box)
        if re.match(r"\d{2}/\d{2}/\d{4}", texto):
            if linha_atual:
                dados.append(linha_atual)
            linha_atual = [texto]
        elif re.match(r"\d{2}:\d{2}", texto):
            linha_atual.append(texto)

    if linha_atual:
        dados.append(linha_atual)

    df = pd.DataFrame(dados, columns=["Data","Entrada1","Saida1","Entrada2","Saida2"])
    return df

def carregar_imagem(file):
    if file.type == "application/pdf":
        pages = convert_from_bytes(file.read())
        return pages
    else:
        return [Image.open(file)]

if uploaded_file:
    paginas = carregar_imagem(uploaded_file)

    dfs = []
    for pagina in paginas:
        df = processar_pagina(pagina)
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    st.subheader("Tabela reconhecida (editável)")
    editado = st.data_editor(final_df, num_rows="dynamic", use_container_width=True)

    csv = editado.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar CSV no modelo", csv, "cartao_ponto.csv", "text/csv")
