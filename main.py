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

st.set_page_config(
    page_title="Leitor de Cartão de Ponto",
    page_icon="🕒",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 96%;
}
.caixa {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 14px;
}
.alerta-box {
    background: #3a1010;
    border: 1px solid #7f1d1d;
    color: #fecaca;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 14px;
}
.sucesso-box {
    background: #052e16;
    border: 1px solid #166534;
    color: #bbf7d0;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("🕒 Leitor de Cartão de Ponto")
st.caption("Leia o cartão, revise os dias e horários, corrija o que for necessário e exporte no modelo exato.")

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
    if not data_str or data_str.lower() == "none":
        return ""

    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return pd.to_datetime(data_str, format=fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return data_str

def normalizar_hora(hora_str):
    hora_str = str(hora_str).strip()
    if not hora_str or hora_str.lower() == "none":
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

def preparar_df_exportacao(df):
    df = df.copy()

    for col in COLUNAS_MODELO:
        if col not in df.columns:
            df[col] = ""

    df = df[COLUNAS_MODELO].copy()

    for col in df.columns:
        df[col] = df[col].fillna("").astype(str).replace("None", "")

    return df

def obter_horarios_sem_par_row(row):
    problemas = []

    for i in range(1, 7):
        entrada_col = f"Entrada{i}"
        saida_col = f"Saída{i}"

        entrada = str(row.get(entrada_col, "")).strip()
        saida = str(row.get(saida_col, "")).strip()

        if entrada and not saida:
            problemas.append(f"{entrada_col}: {entrada}")

        if saida and not entrada:
            problemas.append(f"{saida_col}: {saida}")

    return problemas

def linha_tem_pendencia(row):
    return len(obter_horarios_sem_par_row(row)) > 0

def validar_e_marcar(df_atual, df_original):
    df = preparar_df_exportacao(df_atual).copy()

    status_list = []
    horarios_sem_par_list = []

    for _, row in df.iterrows():
        problemas = obter_horarios_sem_par_row(row)

        if problemas:
            status = "⚠️ Pendente"
            horarios_sem_par = " | ".join(problemas)
        else:
            data_atual = str(row.get("Data", "")).strip()
            corrigido = False

            if data_atual and not df_original.empty:
                orig_mesma_data = df_original[df_original["Data"].astype(str).str.strip() == data_atual]
                for _, row_orig in orig_mesma_data.iterrows():
                    if linha_tem_pendencia(row_orig):
                        corrigido = True
                        break

            status = "✅ Corrigido" if corrigido else ""
            horarios_sem_par = ""

        status_list.append(status)
        horarios_sem_par_list.append(horarios_sem_par)

    df["Status"] = status_list
    df["Horários sem par"] = horarios_sem_par_list
    return df

def gerar_relatorio_pendentes(df):
    linhas = []
    for _, row in df.iterrows():
        if str(row.get("Status", "")).strip() == "⚠️ Pendente":
            linhas.append({
                "Data": row.get("Data", ""),
                "Horários sem par": row.get("Horários sem par", "")
            })
    return pd.DataFrame(linhas)

def gerar_relatorio_corrigidos(df):
    linhas = []
    for _, row in df.iterrows():
        if str(row.get("Status", "")).strip() == "✅ Corrigido":
            linhas.append({
                "Data": row.get("Data", ""),
                "Status": "Corrigido"
            })
    return pd.DataFrame(linhas)

def obter_hash_arquivo(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

if "arquivo_hash" not in st.session_state:
    st.session_state.arquivo_hash = None

if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

if "df_base" not in st.session_state:
    st.session_state.df_base = pd.DataFrame(columns=COLUNAS_MODELO)

if "df_editado" not in st.session_state:
    st.session_state.df_editado = pd.DataFrame(columns=COLUNAS_MODELO)

st.markdown('<div class="caixa">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Envie o cartão de ponto",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Aceita PDF, JPG, JPEG e PNG."
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash = obter_hash_arquivo(file_bytes)
    arquivo_novo = st.session_state.arquivo_hash != file_hash

    if arquivo_novo:
        with st.spinner("Lendo e organizando o cartão..."):
            raw_text = extract_text(file_bytes, uploaded_file.type)
            df_lido = parse_cartao(raw_text)

        st.session_state.arquivo_hash = file_hash
        st.session_state.raw_text = raw_text
        st.session_state.df_base = preparar_df_exportacao(df_lido)
        st.session_state.df_editado = preparar_df_exportacao(df_lido)

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🔄 Reprocessar arquivo", use_container_width=True):
            with st.spinner("Relendo o cartão..."):
                raw_text = extract_text(file_bytes, uploaded_file.type)
                df_lido = parse_cartao(raw_text)

            st.session_state.raw_text = raw_text
            st.session_state.df_base = preparar_df_exportacao(df_lido)
            st.session_state.df_editado = preparar_df_exportacao(df_lido)
            st.rerun()

    with col2:
        csv_data = preparar_df_exportacao(st.session_state.df_editado).to_csv(
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
        st.text(st.session_state.raw_text[:5000] if st.session_state.raw_text else "")

    st.subheader("Tabela para revisão")

    df_exibicao = validar_e_marcar(
        st.session_state.df_editado,
        st.session_state.df_base
    )

    pendentes_df = gerar_relatorio_pendentes(df_exibicao)
    corrigidos_df = gerar_relatorio_corrigidos(df_exibicao)

    if not pendentes_df.empty:
        st.markdown(
            f'<div class="alerta-box"><strong>Atenção:</strong> há {len(pendentes_df)} dia(s) com horário sem par.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="sucesso-box"><strong>Ok:</strong> não há horários sem par pendentes.</div>',
            unsafe_allow_html=True
        )

    if not corrigidos_df.empty:
        st.markdown(
            f'<div class="sucesso-box"><strong>Corrigidos:</strong> {len(corrigidos_df)} dia(s) já foram ajustados.</div>',
            unsafe_allow_html=True
        )

    edited_df = st.data_editor(
        df_exibicao,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="editor_cartao_principal",
        disabled=["Status", "Horários sem par"],
        column_config={
            "Data": st.column_config.TextColumn("Data", width="medium"),
            "Horários sem par": st.column_config.TextColumn("Horários sem par", width="large"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
        }
    )

    edited_df = pd.DataFrame(edited_df)

    if "Status" in edited_df.columns:
        edited_df = edited_df.drop(columns=["Status"])
    if "Horários sem par" in edited_df.columns:
        edited_df = edited_df.drop(columns=["Horários sem par"])

    st.session_state.df_editado = preparar_df_exportacao(edited_df)

    df_final = validar_e_marcar(
        st.session_state.df_editado,
        st.session_state.df_base
    )

    pendentes_df = gerar_relatorio_pendentes(df_final)
    corrigidos_df = gerar_relatorio_corrigidos(df_final)

    st.subheader("Pendências atuais")
    if not pendentes_df.empty:
        st.dataframe(pendentes_df, use_container_width=True, hide_index=True)
    else:
        st.success("Nenhum dia com horário sem par pendente.")

    st.subheader("Dias corrigidos")
    if not corrigidos_df.empty:
        st.dataframe(corrigidos_df, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum dia corrigido até o momento.")

else:
    st.info("Envie um arquivo para iniciar a leitura do cartão.")
