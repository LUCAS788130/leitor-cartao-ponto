import io
import re
import hashlib
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import pdfplumber
import pytesseract
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes


st.set_page_config(
    page_title="Leitor de Cartão de Ponto",
    page_icon="🕒",
    layout="wide",
)

st.markdown("""
<style>
.block-container {
    max-width: 96%;
    padding-top: 1rem;
    padding-bottom: 2rem;
}
.caixa {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 14px;
}
.muted {
    color: #94a3b8;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🕒 Leitor de Cartão de Ponto")
st.caption("Envie o cartão, revise as marcações e exporte no modelo exato.")

COLUNAS_MODELO = [
    "Data",
    "Entrada1", "Saída1",
    "Entrada2", "Saída2",
    "Entrada3", "Saída3",
    "Entrada4", "Saída4",
    "Entrada5", "Saída5",
    "Entrada6", "Saída6",
]

PALAVRAS_STATUS_SEM_MARCACAO = [
    "FERIAS", "FÉRIAS", "DSR", "FERIADO",
    "LIBERAÇÃO", "LIBERACAO",
    "TERMINO DE PRODUÇÃO ANTECIPADA", "TÉRMINO DE PRODUÇÃO ANTECIPADA",
    "COMPENSADO", "COMPENSADO BH",
    "LICENÇA", "LICENCA",
    "NÃO ATIVO", "NAO ATIVO",
    "ABONO", "FOLGA"
]

if "arquivo_hash" not in st.session_state:
    st.session_state.arquivo_hash = None

if "ocr_bruto" not in st.session_state:
    st.session_state.ocr_bruto = ""

if "modelo_detectado" not in st.session_state:
    st.session_state.modelo_detectado = ""

if "df_editado" not in st.session_state:
    st.session_state.df_editado = pd.DataFrame(columns=COLUNAS_MODELO)


def hash_arquivo(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def preparar_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in COLUNAS_MODELO:
        if col not in df.columns:
            df[col] = ""

    df = df[COLUNAS_MODELO].copy()

    for col in COLUNAS_MODELO:
        df[col] = df[col].fillna("").astype(str)
        df[col] = df[col].replace({"None": "", "nan": ""})

    return df


def normalizar_hora(valor: str) -> str:
    valor = str(valor).strip()
    if not valor or valor.lower() in {"none", "nan"}:
        return ""

    m = re.match(r"^(\d{1,2}):(\d{2})$", valor)
    if m:
        return f"{m.group(1).zfill(2)}:{m.group(2)}"

    return valor


def montar_linha(data: str, horarios: list[str]) -> dict:
    linha = {col: "" for col in COLUNAS_MODELO}
    linha["Data"] = data

    horarios = [normalizar_hora(h) for h in horarios[:12]]

    for i, hora in enumerate(horarios, start=1):
        par = (i + 1) // 2
        if i % 2 == 1:
            linha[f"Entrada{par}"] = hora
        else:
            linha[f"Saída{par}"] = hora

    return linha


def preprocess_image(pil_img: Image.Image) -> Image.Image:
    img = np.array(pil_img)

    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    return Image.fromarray(gray)


def best_rotation_for_image(pil_img: Image.Image) -> Image.Image:
    candidates = []
    for angle in [0, 90, 180, 270]:
        img_rot = pil_img.rotate(angle, expand=True)
        txt = pytesseract.image_to_string(preprocess_image(img_rot), lang="por", config="--psm 6")
        score = 0
        score += len(re.findall(r"\b\d{2}/\d{2}/\d{4}\b", txt)) * 5
        score += len(re.findall(r"\b\d{2}:\d{2}\b", txt))
        t = txt.upper()
        for kw in ["LISTAGEM DE MOVIMENTOS", "CARTAO PONTO", "CARTÃO PONTO", "RELATORIO DO CARTAO DE PONTO", "RELATÓRIO DO CARTÃO DE PONTO"]:
            if kw in t:
                score += 20
        candidates.append((score, img_rot))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def images_from_upload(file_bytes: bytes, file_type: str) -> list[Image.Image]:
    if file_type == "application/pdf":
        return convert_from_bytes(file_bytes, dpi=170)
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


def detect_model_from_text(text: str) -> str:
    t = text.upper()

    if "LISTAGEM DE MOVIMENTOS DA FREQUENCIA" in t:
        return "TEXTO_LISTAGEM"

    if "CARTAO PONTO" in t or "CARTÃO PONTO" in t:
        if "MARCACOES" in t or "MARCAÇÕES" in t:
            return "BMG"

    if "RELATORIO DO CARTAO DE PONTO" in t or "RELATÓRIO DO CARTÃO DE PONTO" in t:
        return "ROTACIONADO_RELATORIO"

    return "DESCONHECIDO"


def extract_text_from_pdf_for_detection(file_bytes: bytes) -> str:
    parts = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages[:3]:
                parts.append(page.extract_text() or "")
    except Exception:
        pass
    return "\n".join(parts)


def parse_text_listagem(raw_text: str) -> pd.DataFrame:
    linhas = [l.rstrip() for l in raw_text.splitlines() if l.strip()]
    padrao_data = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")

    registros = []

    for linha in linhas:
        if not padrao_data.search(linha):
            continue

        data = padrao_data.search(linha).group()
        upper = linha.upper()

        # se houver ocorrência tipo FOLGA, mantém o dia vazio
        if any(p in upper for p in PALAVRAS_STATUS_SEM_MARCACAO):
            registros.append(montar_linha(data, []))
            continue

        # pega só o trecho ANTES de HTRAB / ocorrência
        corte = re.split(r"\bHTRAB\b|\bQV\b|\bOCORRENCIA\b", linha, maxsplit=1, flags=re.IGNORECASE)[0]

        # remove data e dia da semana do começo
        corte = re.sub(r"^\s*\d{2}/\d{2}/\d{4}\s+[A-ZÁÉÍÓÚÇ]{3}\s*", "", corte, flags=re.IGNORECASE)

        horas = re.findall(r"\b\d{2}:\d{2}\b", corte)

        # ESTE É O PONTO PRINCIPAL:
        # só usa as 4 primeiras marcações reais da linha
        horas = horas[:4]

        registros.append(montar_linha(data, horas))

    if not registros:
        return pd.DataFrame(columns=COLUNAS_MODELO)

    return preparar_df(pd.DataFrame(registros))


def extract_period_from_bmg_text(text: str):
    m = re.search(r"PER[ÍI]ODO\s*:?\s*(\d{2}/\d{2}/\d{4})\s*A\s*(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
    if m:
        ini = datetime.strptime(m.group(1), "%d/%m/%Y")
        fim = datetime.strptime(m.group(2), "%d/%m/%Y")
        return ini, fim

    m1 = re.search(r"PER[ÍI]ODO\s*:?\s*(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
    m2 = re.search(r"CART[ÃA]O PONTO\s*(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
    if m1 and m2:
        ini = datetime.strptime(m1.group(1), "%d/%m/%Y")
        fim = datetime.strptime(m2.group(1), "%d/%m/%Y")
        return ini, fim

    return None


def intervalo_datas(inicio: datetime, fim: datetime):
    datas = []
    atual = inicio
    while atual <= fim:
        datas.append(atual)
        atual += timedelta(days=1)
    return datas


def extract_bmg_rows_from_text(raw_text: str):
    linhas = [re.sub(r"\s+", " ", l).strip() for l in raw_text.splitlines() if l.strip()]
    rows = []
    for linha in linhas:
        if re.match(r"^\d{2}\s+[A-ZÁÉÍÓÚÇ]{3}\s+\d{4}\b", linha.upper()):
            rows.append(linha)
    return rows


def parse_bmg_row_times(line: str):
    upper = line.upper()

    if any(p in upper for p in PALAVRAS_STATUS_SEM_MARCACAO):
        return []

    m = re.match(r"^\d{2}\s+[A-ZÁÉÍÓÚÇ]{3}\s+\d{4}\s+(.*)$", line, re.IGNORECASE)
    tail = m.group(1) if m else line

    m_mark = re.match(
        r"^((?:\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2})(?:\s*\|\s*(?:\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}))*)",
        tail
    )
    if m_mark:
        return re.findall(r"\b\d{1,2}:\d{2}\b", m_mark.group(1))

    m_short = re.match(r"^(\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2})", tail)
    if m_short:
        return re.findall(r"\b\d{1,2}:\d{2}\b", m_short.group(1))

    return []


def parse_bmg(raw_text: str) -> pd.DataFrame:
    period = extract_period_from_bmg_text(raw_text)
    rows = extract_bmg_rows_from_text(raw_text)

    if not rows:
        return pd.DataFrame(columns=COLUNAS_MODELO)

    registros = []
    if period:
        start_date, end_date = period
        expected_days = intervalo_datas(start_date, end_date)

        line_by_day = {}
        for row in rows:
            m = re.match(r"^(\d{2})\s+", row)
            if m:
                day = int(m.group(1))
                line_by_day[day] = row

        for dt in expected_days:
            row = line_by_day.get(dt.day)
            horas = parse_bmg_row_times(row) if row else []
            registros.append(montar_linha(dt.strftime("%d/%m/%Y"), horas))
    else:
        for row in rows:
            m = re.match(r"^(\d{2})\s+", row)
            if not m:
                continue
            day = m.group(1)
            horas = parse_bmg_row_times(row)
            registros.append(montar_linha(day, horas))

    return preparar_df(pd.DataFrame(registros))


def parse_rotated_report_pages(images: list[Image.Image]) -> tuple[pd.DataFrame, str]:
    textos = []
    registros = []

    for page_img in images:
        img = best_rotation_for_image(page_img)
        w, h = img.size
        img = img.crop((int(w * 0.06), int(h * 0.08), int(w * 0.96), int(h * 0.88)))
        text = pytesseract.image_to_string(preprocess_image(img), lang="por", config="--psm 6")
        textos.append(text)

        linhas = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip()]
        for linha in linhas:
            m = re.search(r"\b\d{2}/\d{2}/\d{4}\b", linha)
            if not m:
                continue

            data = m.group()
            horas = re.findall(r"\b\d{1,2}:\d{2}\b", linha)
            registros.append(montar_linha(data, horas[:6]))

    df = preparar_df(pd.DataFrame(registros)) if registros else pd.DataFrame(columns=COLUNAS_MODELO)
    return df, "\n".join(textos)


def process_file(file_bytes: bytes, file_type: str):
    images = images_from_upload(file_bytes, file_type)

    detection_text = ""
    if file_type == "application/pdf":
        detection_text = extract_text_from_pdf_for_detection(file_bytes)

    model = detect_model_from_text(detection_text)

    if model == "TEXTO_LISTAGEM":
        df = parse_text_listagem(detection_text)
        return model, df, detection_text

    if model == "BMG":
        df = parse_bmg(detection_text)
        return model, df, detection_text

    if model == "DESCONHECIDO" and images:
        first_text = pytesseract.image_to_string(preprocess_image(best_rotation_for_image(images[0])), lang="por", config="--psm 6")
        model = detect_model_from_text(first_text)
        detection_text = first_text

    if model == "ROTACIONADO_RELATORIO":
        df, raw = parse_rotated_report_pages(images)
        return model, df, raw

    if model == "BMG":
        raw_pages = []
        for img in images:
            text = pytesseract.image_to_string(preprocess_image(best_rotation_for_image(img)), lang="por", config="--psm 6")
            raw_pages.append(text)
        full_text = "\n".join(raw_pages)
        df = parse_bmg(full_text)
        return model, df, full_text

    if model == "TEXTO_LISTAGEM":
        raw_pages = []
        for img in images:
            text = pytesseract.image_to_string(preprocess_image(best_rotation_for_image(img)), lang="por", config="--psm 6")
            raw_pages.append(text)
        full_text = "\n".join(raw_pages)
        df = parse_text_listagem(full_text)
        return model, df, full_text

    raw_pages = []
    for img in images:
        text = pytesseract.image_to_string(preprocess_image(best_rotation_for_image(img)), lang="por", config="--psm 6")
        raw_pages.append(text)
    full_text = "\n".join(raw_pages)

    df = parse_text_listagem(full_text)
    return "FALLBACK", df, full_text


st.markdown('<div class="caixa">', unsafe_allow_html=True)
arquivo = st.file_uploader(
    "Envie o cartão de ponto",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Aceita PDF, PNG, JPG e JPEG."
)
st.markdown(
    '<div class="muted">Nesta versão, a validação instantânea foi removida para evitar travamentos.</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

if arquivo is None:
    st.info("Envie um arquivo para iniciar.")
    st.stop()

file_bytes = arquivo.read()
file_hash = hash_arquivo(file_bytes)
arquivo_novo = st.session_state.arquivo_hash != file_hash

if arquivo_novo:
    try:
        with st.spinner("Lendo e organizando o cartão..."):
            modelo, df_lido, texto = process_file(file_bytes, arquivo.type)

        st.session_state.arquivo_hash = file_hash
        st.session_state.ocr_bruto = texto
        st.session_state.modelo_detectado = modelo
        st.session_state.df_editado = preparar_df(df_lido)

    except Exception as e:
        st.error("Erro ao ler o arquivo.")
        st.exception(e)
        st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🔄 Reprocessar arquivo", use_container_width=True):
        try:
            with st.spinner("Relendo o cartão..."):
                modelo, df_lido, texto = process_file(file_bytes, arquivo.type)

            st.session_state.ocr_bruto = texto
            st.session_state.modelo_detectado = modelo
            st.session_state.df_editado = preparar_df(df_lido)
            st.rerun()

        except Exception as e:
            st.error("Erro ao reprocessar o arquivo.")
            st.exception(e)

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

st.write(f"**Modelo detectado:** `{st.session_state.modelo_detectado}`")

with st.expander("Visualizar OCR / texto bruto"):
    st.text(st.session_state.ocr_bruto[:10000] if st.session_state.ocr_bruto else "")

st.subheader("Tabela para revisão")

edited_df = st.data_editor(
    st.session_state.df_editado,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key="editor_principal",
    column_config={
        "Data": st.column_config.TextColumn("Data", width="medium"),
    }
)

st.session_state.df_editado = preparar_df(pd.DataFrame(edited_df))
