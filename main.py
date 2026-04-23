import streamlit as st
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import numpy as np
import re

st.set_page_config(layout="wide")
st.title("Leitor de Cartão de Ponto")

uploaded_file = st.file_uploader(
    "Envie o cartão de ponto (PDF, imagem ou digitalizado)",
    type=["pdf", "png", "jpg", "jpeg"]
)

COLUNAS_MODELO = [
    "Data",
    "Entrada1", "Saída1",
    "Entrada2", "Saída2",
    "Entrada3", "Saída3",
    "Entrada4", "Saída4",
    "Entrada5", "Saída5",
    "Entrada6", "Saída6",
]

def preprocess_image(pil_img):
    img = np.array(pil_img)

    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    return Image.fromarray(thresh)

def ocr_image(image):
    configs = [
        "--psm 6",
        "--psm 11",
    ]

    textos = []
    for cfg in configs:
        try:
            txt = pytesseract.image_to_string(image, lang="por", config=cfg)
        except Exception:
            txt = pytesseract.image_to_string(image, config=cfg)
        textos.append(txt)

    return "\n".join(textos)

def extract_text(file):
    text_parts = []

    if file.type == "application/pdf":
        pdf_bytes = file.read()
        pages = convert_from_bytes(pdf_bytes, dpi=300)

        for page in pages:
            processed = preprocess_image(page)
            text_parts.append(ocr_image(processed))
    else:
        image = Image.open(file).convert("RGB")
        processed = preprocess_image(image)
        text_parts.append(ocr_image(processed))

    return "\n".join(text_parts)

def normalizar_data(data_str):
    data_str = data_str.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return pd.to_datetime(data_str, format=fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return data_str

def normalizar_hora(hora_str):
    hora_str = hora_str.strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", hora_str)
    if m:
        hh = m.group(1).zfill(2)
        mm = m.group(2)
        return f"{hh}:{mm}"
    return hora_str

def montar_linha(data, horarios):
    linha = {col: "" for col in COLUNAS_MODELO}
    linha["Data"] = normalizar_data(data)

    horarios = [normalizar_hora(h) for h in horarios[:12]]

    for i, hora in enumerate(horarios, start=1):
        if i % 2 == 1:
            idx = (i + 1) // 2
            linha[f"Entrada{idx}"] = hora
        else:
            idx = i // 2
            linha[f"Saída{idx}"] = hora

    return linha

def parse_cartao(text):
    linhas = [l.strip() for l in text.splitlines() if l.strip()]

    padrao_data = re.compile(r"\b\d{2}/\d{2}/(?:\d{2}|\d{4})\b")
    padrao_hora = re.compile(r"\b\d{1,2}:\d{2}\b")

    registros = []
    data_atual = None
    horarios_atuais = []

    for linha in linhas:
        datas_encontradas = padrao_data.findall(linha)
        horas_encontradas = padrao_hora.findall(linha)

        if datas_encontradas:
            # Salva o bloco anterior SEMPRE, mesmo sem horários
            if data_atual is not None:
                registros.append(montar_linha(data_atual, horarios_atuais))

            data_atual = datas_encontradas[0]
            horarios_atuais = horas_encontradas.copy()
        else:
            if data_atual is not None and horas_encontradas:
                horarios_atuais.extend(horas_encontradas)

    # Salva o último dia SEMPRE, mesmo sem horários
    if data_atual is not None:
        registros.append(montar_linha(data_atual, horarios_atuais))

    if not registros:
        return pd.DataFrame(columns=COLUNAS_MODELO)

    df = pd.DataFrame(registros)

    for col in COLUNAS_MODELO:
        if col not in df.columns:
            df[col] = ""

    return df[COLUNAS_MODELO]

def adicionar_linha_vazia(df):
    nova = pd.DataFrame([{col: "" for col in COLUNAS_MODELO}])
    return pd.concat([df, nova], ignore_index=True)

if uploaded_file:
    st.info("Lendo o cartão...")

    raw_text = extract_text(uploaded_file)
    df = parse_cartao(raw_text)

    if "df_editado" not in st.session_state:
        st.session_state.df_editado = df.copy()
    else:
        st.session_state.df_editado = df.copy()

    with st.expander("Mostrar OCR bruto"):
        st.text(raw_text[:15000] if raw_text else "")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Adicionar linha vazia"):
            st.session_state.df_editado = adicionar_linha_vazia(st.session_state.df_editado)

    with col2:
        if st.button("Recarregar leitura do arquivo"):
            st.session_state.df_editado = df.copy()

    st.subheader("Tabela final no modelo exato")

    edited_df = st.data_editor(
        st.session_state.df_editado,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="editor_cartao"
    )

    st.session_state.df_editado = edited_df.copy()

    csv_data = edited_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    st.download_button(
        "Baixar CSV",
        data=csv_data,
        file_name="cartao_ponto_modelo_exato.csv",
        mime="text/csv"
    )
