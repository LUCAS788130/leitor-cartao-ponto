import streamlit as st
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import numpy as np
import re
import hashlib
import io

st.set_page_config(page_title="Leitor de Cartão de Ponto", page_icon="🕒", layout="wide")

COLUNAS_MODELO = [
    "Data",
    "Entrada1", "Saída1",
    "Entrada2", "Saída2",
    "Entrada3", "Saída3",
    "Entrada4", "Saída4",
    "Entrada5", "Saída5",
    "Entrada6", "Saída6",
]

st.title("🕒 Leitor de Cartão de Ponto")
st.caption("Envie o cartão, revise os horários e exporte no modelo exato.")

def preprocess_image(pil_img):
    img = np.array(pil_img)
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    return Image.fromarray(gray)

def ocr_image(image):
    try:
        return pytesseract.image_to_string(image, lang="por", config="--psm 6")
    except Exception:
        return pytesseract.image_to_string(image, config="--psm 6")

def extract_text(file_bytes, file_type):
    partes = []

    if file_type == "application/pdf":
        paginas = convert_from_bytes(file_bytes, dpi=150)
        for pagina in paginas:
            processada = preprocess_image(pagina)
            partes.append(ocr_image(processada))
    else:
        imagem = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        processada = preprocess_image(imagem)
        partes.append(ocr_image(processada))

    return "\n".join(partes)

def normalizar_data(valor):
    valor = str(valor).strip()
    if not valor or valor.lower() == "none":
        return ""
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return pd.to_datetime(valor, format=fmt).strftime("%d/%m/%Y")
        except ValueError:
            pass
    return valor

def normalizar_hora(valor):
    valor = str(valor).strip()
    if not valor or valor.lower() == "none":
        return ""
    m = re.match(r"^(\d{1,2}):(\d{2})$", valor)
    if m:
        return f"{m.group(1).zfill(2)}:{m.group(2)}"
    return valor

def montar_linha(data, horarios):
    linha = {col: "" for col in COLUNAS_MODELO}
    linha["Data"] = normalizar_data(data)

    horarios = [normalizar_hora(h) for h in horarios[:12]]
    for i, hora in enumerate(horarios, start=1):
        par = (i + 1) // 2
        if i % 2 == 1:
            linha[f"Entrada{par}"] = hora
        else:
            linha[f"Saída{par}"] = hora
    return linha

def parse_cartao(texto):
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    padrao_data = re.compile(r"\b\d{2}/\d{2}/(?:\d{2}|\d{4})\b")
    padrao_hora = re.compile(r"\b\d{1,2}:\d{2}\b")

    registros = []
    data_atual = None
    horarios_atuais = []

    for linha in linhas:
        datas = padrao_data.findall(linha)
        horas = padrao_hora.findall(linha)

        if datas:
            if data_atual is not None:
                registros.append(montar_linha(data_atual, horarios_atuais))
            data_atual = datas[0]
            horarios_atuais = horas.copy()
        elif data_atual is not None and horas:
            horarios_atuais.extend(horas)

    if data_atual is not None:
        registros.append(montar_linha(data_atual, horarios_atuais))

    if not registros:
        return pd.DataFrame(columns=COLUNAS_MODELO)

    df = pd.DataFrame(registros)
    for col in COLUNAS_MODELO:
        if col not in df.columns:
            df[col] = ""
    return df[COLUNAS_MODELO]

def preparar_df(df):
    df = df.copy()
    for col in COLUNAS_MODELO:
        if col not in df.columns:
            df[col] = ""
    df = df[COLUNAS_MODELO].copy()

    for col in COLUNAS_MODELO:
        df[col] = df[col].fillna("").astype(str).replace("None", "")
        if col == "Data":
            df[col] = df[col].apply(normalizar_data)
        else:
            df[col] = df[col].apply(normalizar_hora)
    return df

def horarios_sem_par(row):
    problemas = []
    for i in range(1, 7):
        ent = str(row.get(f"Entrada{i}", "")).strip()
        sai = str(row.get(f"Saída{i}", "")).strip()
        if ent and not sai:
            problemas.append(f"Entrada{i}: {ent}")
        if sai and not ent:
            problemas.append(f"Saída{i}: {sai}")
    return problemas

def validar_df(df):
    df = preparar_df(df)
    status = []
    pendencias = []

    for _, row in df.iterrows():
        probs = horarios_sem_par(row)
        if probs:
            status.append("⚠️ Pendente")
            pendencias.append(" | ".join(probs))
        else:
            status.append("")
            pendencias.append("")

    df["Status"] = status
    df["Horários sem par"] = pendencias
    return df

def hash_arquivo(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

if "arquivo_hash" not in st.session_state:
    st.session_state.arquivo_hash = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "df_editado" not in st.session_state:
    st.session_state.df_editado = pd.DataFrame(columns=COLUNAS_MODELO)

uploaded_file = st.file_uploader(
    "Envie o cartão de ponto",
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file:
    try:
        file_bytes = uploaded_file.read()
        file_hash = hash_arquivo(file_bytes)
        arquivo_novo = st.session_state.arquivo_hash != file_hash

        if arquivo_novo:
            with st.spinner("Lendo e organizando o cartão..."):
                raw_text = extract_text(file_bytes, uploaded_file.type)
                df_lido = parse_cartao(raw_text)
                df_lido = preparar_df(df_lido)

            st.session_state.arquivo_hash = file_hash
            st.session_state.raw_text = raw_text
            st.session_state.df_editado = df_lido.copy()

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔄 Reprocessar arquivo", use_container_width=True):
                with st.spinner("Relendo o cartão..."):
                    raw_text = extract_text(file_bytes, uploaded_file.type)
                    df_lido = parse_cartao(raw_text)
                    df_lido = preparar_df(df_lido)

                st.session_state.raw_text = raw_text
                st.session_state.df_editado = df_lido.copy()
                st.rerun()

        with col2:
            csv_data = preparar_df(st.session_state.df_editado).to_csv(
                index=False,
                encoding="utf-8-sig"
            ).encode("utf-8-sig")

            st.download_button(
                "📥 Baixar CSV",
                data=csv_data,
                file_name="cartao_ponto_modelo_exato.csv",
                mime="text/csv",
                use_container_width=True
            )

        with st.expander("Visualizar OCR bruto"):
            st.text(st.session_state.raw_text[:3000] if st.session_state.raw_text else "")

        st.subheader("Tabela para revisão")

        df_exibicao = validar_df(st.session_state.df_editado)

        pendentes = df_exibicao[df_exibicao["Status"] == "⚠️ Pendente"][["Data", "Horários sem par"]]
        if len(pendentes) > 0:
            st.warning(f"Há {len(pendentes)} dia(s) com horário sem par.")
        else:
            st.success("Nenhum dia com horário sem par.")

        edited_df = st.data_editor(
            df_exibicao,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="editor_cartao_principal",
            disabled=["Status", "Horários sem par"]
        )

        edited_df = pd.DataFrame(edited_df)
        for col_extra in ["Status", "Horários sem par"]:
            if col_extra in edited_df.columns:
                edited_df = edited_df.drop(columns=[col_extra])

        st.session_state.df_editado = preparar_df(edited_df)

    except Exception as e:
        st.error("Erro ao processar o cartão.")
        st.exception(e)

else:
    st.info("Envie um arquivo para iniciar.")
