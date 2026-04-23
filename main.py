import streamlit as st
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import numpy as np
import re
import hashlib

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

COLUNAS_EXIBICAO = COLUNAS_MODELO + ["Inconsistência", "Horários sem par"]


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
    try:
        return pytesseract.image_to_string(image, lang="por", config="--psm 6")
    except Exception:
        return pytesseract.image_to_string(image, config="--psm 6")


def extract_text(file_bytes, file_type):
    text_parts = []

    if file_type == "application/pdf":
        pages = convert_from_bytes(file_bytes, dpi=200)
        for page in pages:
            processed = preprocess_image(page)
            text_parts.append(ocr_image(processed))
    else:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        processed = preprocess_image(image)
        text_parts.append(ocr_image(processed))

    return "\n".join(text_parts)


def normalizar_data(data_str):
    data_str = str(data_str).strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return pd.to_datetime(data_str, format=fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return data_str


def normalizar_hora(hora_str):
    hora_str = str(hora_str).strip()
    if not hora_str:
        return ""
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
            if data_atual is not None:
                registros.append(montar_linha(data_atual, horarios_atuais))

            data_atual = datas_encontradas[0]
            horarios_atuais = horas_encontradas.copy()
        else:
            if data_atual is not None and horas_encontradas:
                horarios_atuais.extend(horas_encontradas)

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


def validar_pares(df):
    df = df.copy()

    inconsistencias = []
    horarios_sem_par = []

    for _, row in df.iterrows():
        problemas = []
        horarios_problematicos = []

        for i in range(1, 7):
            entrada_col = f"Entrada{i}"
            saida_col = f"Saída{i}"

            entrada = str(row.get(entrada_col, "")).strip()
            saida = str(row.get(saida_col, "")).strip()

            if entrada and not saida:
                problemas.append(f"{entrada_col} sem {saida_col}")
                horarios_problematicos.append(entrada)

            if saida and not entrada:
                problemas.append(f"{saida_col} sem {entrada_col}")
                horarios_problematicos.append(saida)

        if problemas:
            inconsistencias.append("⚠️ SIM")
            horarios_sem_par.append(" | ".join(horarios_problematicos))
        else:
            inconsistencias.append("")
            horarios_sem_par.append("")

    df["Inconsistência"] = inconsistencias
    df["Horários sem par"] = horarios_sem_par
    return df


def gerar_relatorio_inconsistencias(df):
    linhas = []

    for _, row in df.iterrows():
        data = str(row.get("Data", "")).strip()
        for i in range(1, 7):
            entrada_col = f"Entrada{i}"
            saida_col = f"Saída{i}"

            entrada = str(row.get(entrada_col, "")).strip()
            saida = str(row.get(saida_col, "")).strip()

            if entrada and not saida:
                linhas.append({
                    "Data": data,
                    "Campo": entrada_col,
                    "Horário sem par": entrada,
                    "Problema": f"{entrada_col} preenchida sem {saida_col}"
                })

            if saida and not entrada:
                linhas.append({
                    "Data": data,
                    "Campo": saida_col,
                    "Horário sem par": saida,
                    "Problema": f"{saida_col} preenchida sem {entrada_col}"
                })

    return pd.DataFrame(linhas)


def style_inconsistencias(df):
    def highlight(_):
        return ["background-color: #5c1f1f; color: white; font-weight: bold;"] * len(df.columns)
    return df.style.apply(highlight, axis=1)


def obter_hash_arquivo(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()


# io import local para evitar erro se não houver upload
import io

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash = obter_hash_arquivo(file_bytes)

    if "arquivo_hash" not in st.session_state:
        st.session_state.arquivo_hash = None

    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""

    if "df_base" not in st.session_state:
        st.session_state.df_base = pd.DataFrame(columns=COLUNAS_MODELO)

    if "df_editado" not in st.session_state:
        st.session_state.df_editado = pd.DataFrame(columns=COLUNAS_MODELO)

    arquivo_novo = st.session_state.arquivo_hash != file_hash

    if arquivo_novo:
        with st.spinner("Lendo o cartão..."):
            raw_text = extract_text(file_bytes, uploaded_file.type)
            df_lido = parse_cartao(raw_text)

        st.session_state.arquivo_hash = file_hash
        st.session_state.raw_text = raw_text
        st.session_state.df_base = df_lido.copy()
        st.session_state.df_editado = df_lido.copy()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Adicionar linha vazia"):
            st.session_state.df_editado = adicionar_linha_vazia(st.session_state.df_editado)

    with col2:
        if st.button("Reprocessar arquivo"):
            with st.spinner("Relendo o cartão..."):
                raw_text = extract_text(file_bytes, uploaded_file.type)
                df_lido = parse_cartao(raw_text)

            st.session_state.raw_text = raw_text
            st.session_state.df_base = df_lido.copy()
            st.session_state.df_editado = df_lido.copy()

    with st.expander("Mostrar OCR bruto"):
        st.text(st.session_state.raw_text[:5000] if st.session_state.raw_text else "")

    st.subheader("Tabela final no modelo exato")

    df_para_edicao = validar_pares(st.session_state.df_editado)

    edited_df = st.data_editor(
        df_para_edicao,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="editor_cartao",
        disabled=["Inconsistência", "Horários sem par"]
    )

    # Mantém no estado apenas as colunas reais do CSV
    st.session_state.df_editado = edited_df[COLUNAS_MODELO].copy()

    df_validado = validar_pares(st.session_state.df_editado)
    df_inconsistencias = gerar_relatorio_inconsistencias(st.session_state.df_editado)

    total_inconsistencias = len(df_inconsistencias)

    if total_inconsistencias > 0:
        st.error(f"Foram encontrados {total_inconsistencias} horário(s) sem par.")

        st.subheader("Dias e horários com inconsistência")
        st.dataframe(
            style_inconsistencias(df_inconsistencias),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("Nenhum horário sem par encontrado.")

    csv_data = st.session_state.df_editado.to_csv(
        index=False,
        encoding="utf-8-sig"
    ).encode("utf-8-sig")

    st.download_button(
        "Baixar CSV",
        data=csv_data,
        file_name="cartao_ponto_modelo_exato.csv",
        mime="text/csv"
    )
