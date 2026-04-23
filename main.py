import io
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pdfplumber
import pytesseract
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes


# =========================================================
# CONFIGURAÇÃO
# =========================================================

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
.alerta-vermelho {
    background: #3a1010;
    border: 1px solid #7f1d1d;
    color: #fecaca;
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 14px;
}
.alerta-verde {
    background: #052e16;
    border: 1px solid #166534;
    color: #bbf7d0;
    border-radius: 12px;
    padding: 12px 14px;
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

COLUNAS_AUX = ["Status", "Horários sem par"]

PALAVRAS_STATUS_SEM_MARCACAO = [
    "FERIAS",
    "FÉRIAS",
    "DSR",
    "FERIADO",
    "LIBERAÇÃO",
    "LIBERACAO",
    "TERMINO DE PRODUÇÃO ANTECIPADA",
    "TÉRMINO DE PRODUÇÃO ANTECIPADA",
    "COMPENSADO",
    "COMPENSADO BH",
    "LICENÇA",
    "LICENCA",
    "NÃO ATIVO",
    "NAO ATIVO",
    "ABONO",
]

# =========================================================
# ESTADO
# =========================================================

if "arquivo_hash" not in st.session_state:
    st.session_state.arquivo_hash = None

if "ocr_bruto" not in st.session_state:
    st.session_state.ocr_bruto = ""

if "modelo_detectado" not in st.session_state:
    st.session_state.modelo_detectado = ""

if "df_original" not in st.session_state:
    st.session_state.df_original = pd.DataFrame(columns=COLUNAS_MODELO)

if "df_editado" not in st.session_state:
    st.session_state.df_editado = pd.DataFrame(columns=COLUNAS_MODELO)


# =========================================================
# ESTRUTURAS
# =========================================================

@dataclass
class ParseResult:
    model_name: str
    df: pd.DataFrame
    raw_text: str


# =========================================================
# UTILITÁRIOS BÁSICOS
# =========================================================

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
        hh = m.group(1).zfill(2)
        mm = m.group(2)
        return f"{hh}:{mm}"

    return valor


def montar_linha(data: str, horarios: List[str]) -> dict:
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


def extrair_horas(texto: str) -> List[str]:
    return re.findall(r"\b\d{1,2}:\d{2}\b", texto)


def limpar_texto(texto: str) -> str:
    texto = texto.replace("\u00A0", " ")
    texto = texto.replace("—", "-")
    texto = texto.replace("–", "-")
    texto = re.sub(r"[|]+", " | ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def intervalo_datas(inicio: datetime, fim: datetime) -> List[datetime]:
    datas = []
    atual = inicio
    while atual <= fim:
        datas.append(atual)
        atual += timedelta(days=1)
    return datas


def row_has_any_time(row: pd.Series) -> bool:
    return any(str(row.get(c, "")).strip() for c in COLUNAS_MODELO if c != "Data")


# =========================================================
# OCR / IMAGEM
# =========================================================

def preprocess_image(pil_img: Image.Image) -> Image.Image:
    img = np.array(pil_img)

    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    return Image.fromarray(gray)


def try_osd_rotation(pil_img: Image.Image) -> Optional[int]:
    try:
        osd = pytesseract.image_to_osd(pil_img)
        m = re.search(r"Rotate: (\d+)", osd)
        if m:
            return int(m.group(1))
    except Exception:
        return None
    return None


def rotate_by_angle(pil_img: Image.Image, angle: int) -> Image.Image:
    if angle == 90:
        return pil_img.rotate(-90, expand=True)
    if angle == 180:
        return pil_img.rotate(180, expand=True)
    if angle == 270:
        return pil_img.rotate(90, expand=True)
    return pil_img


def ocr_text_score(text: str) -> int:
    t = text.upper()
    score = 0
    score += len(re.findall(r"\b\d{2}/\d{2}/\d{4}\b", text)) * 5
    score += len(re.findall(r"\b\d{2}:\d{2}\b", text)) * 1
    for kw in [
        "CARTÃO PONTO",
        "CARTAO PONTO",
        "RELATÓRIO DO CARTÃO DE PONTO",
        "RELATORIO DO CARTAO DE PONTO",
        "LISTAGEM DE MOVIMENTOS",
        "MARCAÇÕES",
        "MARCACOES",
        "ENTRA",
        "I.INI",
        "I.FIN",
        "SAIDA",
    ]:
        if kw in t:
            score += 20
    return score


def best_rotation_for_image(pil_img: Image.Image) -> Image.Image:
    # tenta OSD primeiro
    angle = try_osd_rotation(pil_img)
    if angle in {90, 180, 270}:
        return rotate_by_angle(pil_img, angle)

    # fallback heurístico
    candidates = []
    for angle in [0, 90, 180, 270]:
        img_rot = rotate_by_angle(pil_img, angle)
        txt = pytesseract.image_to_string(preprocess_image(img_rot), lang="por", config="--psm 6")
        candidates.append((ocr_text_score(txt), img_rot))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def pdf_pages_to_images(file_bytes: bytes) -> List[Image.Image]:
    return convert_from_bytes(file_bytes, dpi=170)


def images_from_upload(file_bytes: bytes, file_type: str) -> List[Image.Image]:
    if file_type == "application/pdf":
        return pdf_pages_to_images(file_bytes)

    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return [img]


# =========================================================
# DETECÇÃO DE MODELO
# =========================================================

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


# =========================================================
# PARSER 1 - MODELO TEXTO LISTAGEM
# =========================================================

def parse_text_listagem(raw_text: str) -> pd.DataFrame:
    linhas = [l.rstrip() for l in raw_text.splitlines() if l.strip()]
    padrao_data = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
    registros = []

    for linha in linhas:
        if not padrao_data.search(linha):
            continue

        data = padrao_data.search(linha).group()
        horas = extrair_horas(linha)
        registros.append(montar_linha(data, horas))

    if not registros:
        return pd.DataFrame(columns=COLUNAS_MODELO)

    return preparar_df(pd.DataFrame(registros))


# =========================================================
# PARSER 2 - MODELO BMG / CARTÃO PONTO
# =========================================================

def extract_period_from_bmg_text(text: str) -> Optional[Tuple[datetime, datetime]]:
    # tenta padrão "Período : 26/11/2025 a 25/12/2025"
    m = re.search(r"PER[ÍI]ODO\s*:?\s*(\d{2}/\d{2}/\d{4})\s*A\s*(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
    if m:
        ini = datetime.strptime(m.group(1), "%d/%m/%Y")
        fim = datetime.strptime(m.group(2), "%d/%m/%Y")
        return ini, fim

    # variação quebrada em duas partes
    m1 = re.search(r"PER[ÍI]ODO\s*:?\s*(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
    m2 = re.search(r"CART[ÃA]O PONTO\s*(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
    if m1 and m2:
        ini = datetime.strptime(m1.group(1), "%d/%m/%Y")
        fim = datetime.strptime(m2.group(1), "%d/%m/%Y")
        return ini, fim

    return None


def extract_bmg_rows_from_text(raw_text: str) -> List[str]:
    linhas = [limpar_texto(l) for l in raw_text.splitlines() if l.strip()]
    rows = []
    for linha in linhas:
        if re.match(r"^\d{2}\s+[A-ZÁÉÍÓÚÇ]{3}\s+\d{4}\b", linha.upper()):
            rows.append(linha)
    return rows


def parse_bmg_row_times(line: str) -> List[str]:
    upper = line.upper()

    for palavra in PALAVRAS_STATUS_SEM_MARCACAO:
        if palavra in upper:
            # dias como férias / dsr / liberação etc entram sem horários
            return []

    # tira prefixo "26 SEX 0733 "
    m = re.match(r"^\d{2}\s+[A-ZÁÉÍÓÚÇ]{3}\s+\d{4}\s+(.*)$", line, re.IGNORECASE)
    tail = m.group(1) if m else line

    # captura só o bloco inicial de marcações:
    # ex: "05:34 - 09:00 | 10:00 - 11:17"
    m_mark = re.match(
        r"^((?:\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2})(?:\s*\|\s*(?:\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}))*)",
        tail
    )
    if m_mark:
        return extrair_horas(m_mark.group(1))

    # caso de sábado curto: "04:42 - 08:52"
    m_short = re.match(r"^(\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2})", tail)
    if m_short:
        return extrair_horas(m_short.group(1))

    return []


def parse_bmg(raw_text: str) -> pd.DataFrame:
    period = extract_period_from_bmg_text(raw_text)
    rows = extract_bmg_rows_from_text(raw_text)

    if not rows:
        return pd.DataFrame(columns=COLUNAS_MODELO)

    # tenta usar o período para não perder nenhum dia
    registros = []
    if period:
        start_date, end_date = period
        expected_days = intervalo_datas(start_date, end_date)

        # mapa dia -> linha; usa a ordem para não depender só do número
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
        # fallback sem período
        for row in rows:
            m = re.match(r"^(\d{2})\s+", row)
            if not m:
                continue
            day = m.group(1)
            horas = parse_bmg_row_times(row)
            registros.append(montar_linha(day, horas))

    return preparar_df(pd.DataFrame(registros))


# =========================================================
# PARSER 3 - RELATÓRIO ROTACIONADO DIGITALIZADO
# =========================================================

def crop_rotated_report_region(pil_img: Image.Image) -> Image.Image:
    # após rotação correta, remove margens e assinatura
    w, h = pil_img.size
    left = int(w * 0.06)
    right = int(w * 0.96)
    top = int(h * 0.08)
    bottom = int(h * 0.88)
    return pil_img.crop((left, top, right, bottom))


def ocr_page_with_best_rotation(pil_img: Image.Image) -> str:
    img = best_rotation_for_image(pil_img)
    img = crop_rotated_report_region(img)
    return pytesseract.image_to_string(preprocess_image(img), lang="por", config="--psm 6")


def extract_rotated_report_period(text: str) -> Optional[Tuple[datetime, datetime]]:
    # tenta "Mês/Ano: 09/2025", mas preferimos datas completas se houver
    # o relatório costuma listar datas completas na linha
    datas = re.findall(r"\b\d{2}/\d{2}/\d{4}\b", text)
    if not datas:
        return None
    dts = [datetime.strptime(d, "%d/%m/%Y") for d in datas]
    return min(dts), max(dts)


def parse_rotated_report_pages(images: List[Image.Image]) -> ParseResult:
    textos = []
    registros = []

    for page_img in images:
        text = ocr_page_with_best_rotation(page_img)
        textos.append(text)

        linhas = [limpar_texto(l) for l in text.splitlines() if l.strip()]
        for linha in linhas:
            # modelo do relatório gira em torno de linha começando com data completa
            if not re.search(r"\b\d{2}/\d{2}/\d{4}\b", linha):
                continue

            data = re.search(r"\b\d{2}/\d{2}/\d{4}\b", linha).group()
            horas = extrair_horas(linha)

            # geralmente só nos interessam as primeiras 4 ou 6 horas da linha
            registros.append(montar_linha(data, horas))

    df = preparar_df(pd.DataFrame(registros)) if registros else pd.DataFrame(columns=COLUNAS_MODELO)
    raw_text = "\n".join(textos)
    return ParseResult("ROTACIONADO_RELATORIO", df, raw_text)


# =========================================================
# PROCESSAMENTO CENTRAL
# =========================================================

def extract_text_from_pdf_for_detection(file_bytes: bytes) -> str:
    # tenta extrair texto nativo das primeiras páginas
    parts = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages[:3]:
                parts.append(page.extract_text() or "")
    except Exception:
        pass
    return "\n".join(parts)


def process_file(file_bytes: bytes, file_type: str) -> ParseResult:
    images = images_from_upload(file_bytes, file_type)

    detection_text = ""
    if file_type == "application/pdf":
        detection_text = extract_text_from_pdf_for_detection(file_bytes)

    model = detect_model_from_text(detection_text)

    # se o texto nativo já identificou bem
    if model == "TEXTO_LISTAGEM":
        df = parse_text_listagem(detection_text)
        return ParseResult(model, df, detection_text)

    if model == "BMG":
        df = parse_bmg(detection_text)
        return ParseResult(model, df, detection_text)

    # se não identificou pelo texto nativo, tenta OCR da primeira página
    if model == "DESCONHECIDO" and images:
        first_text = ocr_page_with_best_rotation(images[0])
        model = detect_model_from_text(first_text)
        detection_text = first_text

    if model == "ROTACIONADO_RELATORIO":
        return parse_rotated_report_pages(images)

    if model == "BMG":
        # PDF escaneado BMG
        raw_pages = []
        for img in images:
            text = pytesseract.image_to_string(preprocess_image(best_rotation_for_image(img)), lang="por", config="--psm 6")
            raw_pages.append(text)
        full_text = "\n".join(raw_pages)
        df = parse_bmg(full_text)
        return ParseResult(model, df, full_text)

    if model == "TEXTO_LISTAGEM":
        raw_pages = []
        for img in images:
            text = pytesseract.image_to_string(preprocess_image(best_rotation_for_image(img)), lang="por", config="--psm 6")
            raw_pages.append(text)
        full_text = "\n".join(raw_pages)
        df = parse_text_listagem(full_text)
        return ParseResult(model, df, full_text)

    # fallback: OCR geral e tenta BMG ou relatório
    raw_pages = []
    for img in images:
        text = pytesseract.image_to_string(preprocess_image(best_rotation_for_image(img)), lang="por", config="--psm 6")
        raw_pages.append(text)

    full_text = "\n".join(raw_pages)
    fallback_model = detect_model_from_text(full_text)

    if fallback_model == "BMG":
        df = parse_bmg(full_text)
        return ParseResult(fallback_model, df, full_text)

    if fallback_model == "TEXTO_LISTAGEM":
        df = parse_text_listagem(full_text)
        return ParseResult(fallback_model, df, full_text)

    # último fallback: tenta relatório rotacionado
    return parse_rotated_report_pages(images)


# =========================================================
# VALIDAÇÃO
# =========================================================

def horarios_sem_par_row(row: pd.Series) -> List[str]:
    problemas = []

    for i in range(1, 7):
        entrada = str(row.get(f"Entrada{i}", "")).strip()
        saida = str(row.get(f"Saída{i}", "")).strip()

        if entrada and not saida:
            problemas.append(f"Entrada{i}: {entrada}")

        if saida and not entrada:
            problemas.append(f"Saída{i}: {saida}")

    return problemas


def linha_tem_pendencia(row: pd.Series) -> bool:
    return len(horarios_sem_par_row(row)) > 0


def validar_e_marcar(df_atual: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
    df = preparar_df(df_atual)

    status = []
    problemas_txt = []

    for _, row in df.iterrows():
        problemas = horarios_sem_par_row(row)

        if problemas:
            status.append("⚠️ Pendente")
            problemas_txt.append(" | ".join(problemas))
        else:
            data_atual = str(row.get("Data", "")).strip()
            corrigido = False

            if data_atual and not df_original.empty:
                base_mesma_data = df_original[
                    df_original["Data"].astype(str).str.strip() == data_atual
                ]
                for _, row_orig in base_mesma_data.iterrows():
                    if linha_tem_pendencia(row_orig):
                        corrigido = True
                        break

            status.append("✅ Corrigido" if corrigido else "")
            problemas_txt.append("")

    df["Status"] = status
    df["Horários sem par"] = problemas_txt
    return df


def relatorio_pendentes(df_validado: pd.DataFrame) -> pd.DataFrame:
    return df_validado[df_validado["Status"] == "⚠️ Pendente"][["Data", "Horários sem par"]].copy()


def relatorio_corrigidos(df_validado: pd.DataFrame) -> pd.DataFrame:
    return df_validado[df_validado["Status"] == "✅ Corrigido"][["Data", "Status"]].copy()


# =========================================================
# INTERFACE
# =========================================================

st.markdown('<div class="caixa">', unsafe_allow_html=True)
arquivo = st.file_uploader(
    "Envie o cartão de ponto",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Aceita PDF, PNG, JPG e JPEG."
)
st.markdown(
    '<div class="muted">O sistema tenta detectar o modelo automaticamente e você revisa antes de exportar.</div>',
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
            result = process_file(file_bytes, arquivo.type)

        st.session_state.arquivo_hash = file_hash
        st.session_state.ocr_bruto = result.raw_text
        st.session_state.modelo_detectado = result.model_name
        st.session_state.df_original = preparar_df(result.df)
        st.session_state.df_editado = preparar_df(result.df)

    except Exception as e:
        st.error("Erro ao ler o arquivo.")
        st.exception(e)
        st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🔄 Reprocessar arquivo", use_container_width=True):
        try:
            with st.spinner("Relendo o cartão..."):
                result = process_file(file_bytes, arquivo.type)

            st.session_state.ocr_bruto = result.raw_text
            st.session_state.modelo_detectado = result.model_name
            st.session_state.df_original = preparar_df(result.df)
            st.session_state.df_editado = preparar_df(result.df)
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

df_exibicao = validar_e_marcar(
    st.session_state.df_editado,
    st.session_state.df_original
)

pendentes = relatorio_pendentes(df_exibicao)
corrigidos = relatorio_corrigidos(df_exibicao)

if not pendentes.empty:
    st.markdown(
        f'<div class="alerta-vermelho"><strong>Atenção:</strong> há {len(pendentes)} dia(s) com horário sem par.</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="alerta-verde"><strong>Ok:</strong> não há horários sem par pendentes.</div>',
        unsafe_allow_html=True
    )

if not corrigidos.empty:
    st.markdown(
        f'<div class="alerta-verde"><strong>Corrigidos:</strong> {len(corrigidos)} dia(s) já foram ajustados.</div>',
        unsafe_allow_html=True
    )

edited_df = st.data_editor(
    df_exibicao,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key="editor_principal",
    disabled=["Status", "Horários sem par"],
    column_config={
        "Data": st.column_config.TextColumn("Data", width="medium"),
        "Status": st.column_config.TextColumn("Status", width="medium"),
        "Horários sem par": st.column_config.TextColumn("Horários sem par", width="large"),
    }
)

edited_df = pd.DataFrame(edited_df)

for col_extra in COLUNAS_AUX:
    if col_extra in edited_df.columns:
        edited_df = edited_df.drop(columns=[col_extra])

st.session_state.df_editado = preparar_df(edited_df)

df_final = validar_e_marcar(
    st.session_state.df_editado,
    st.session_state.df_original
)

pendentes_finais = relatorio_pendentes(df_final)
corrigidos_finais = relatorio_corrigidos(df_final)

st.subheader("Pendências atuais")
if pendentes_finais.empty:
    st.success("Nenhum dia com horário sem par pendente.")
else:
    st.dataframe(pendentes_finais, use_container_width=True, hide_index=True)

st.subheader("Dias corrigidos")
if corrigidos_finais.empty:
    st.info("Nenhum dia corrigido até o momento.")
else:
    st.dataframe(corrigidos_finais, use_container_width=True, hide_index=True)
