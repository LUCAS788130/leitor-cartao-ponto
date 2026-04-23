import io
import re
import hashlib
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes


# =========================================================
# CONFIGURAÇÃO DA PÁGINA
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
st.caption("Envie o cartão, revise os dias e horários e exporte no modelo exato.")

# =========================================================
# MODELO DE SAÍDA
# =========================================================
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

# =========================================================
# ESTADO
# =========================================================
if "arquivo_hash" not in st.session_state:
    st.session_state.arquivo_hash = None

if "ocr_bruto" not in st.session_state:
    st.session_state.ocr_bruto = ""

if "df_original" not in st.session_state:
    st.session_state.df_original = pd.DataFrame(columns=COLUNAS_MODELO)

if "df_editado" not in st.session_state:
    st.session_state.df_editado = pd.DataFrame(columns=COLUNAS_MODELO)


# =========================================================
# UTILITÁRIOS
# =========================================================
def hash_arquivo(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def normalizar_data(valor: str) -> str:
    valor = str(valor).strip()
    if not valor or valor.lower() == "none":
        return ""

    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%d/%m"):
        try:
            dt = pd.to_datetime(valor, format=fmt)
            if fmt == "%d/%m":
                return dt.strftime("%d/%m")
            return dt.strftime("%d/%m/%Y")
        except ValueError:
            pass

    return valor


def normalizar_hora(valor: str) -> str:
    valor = str(valor).strip()
    if not valor or valor.lower() == "none":
        return ""

    m = re.match(r"^(\d{1,2}):(\d{2})$", valor)
    if m:
        return f"{m.group(1).zfill(2)}:{m.group(2)}"

    return valor


def preparar_df(df: pd.DataFrame) -> pd.DataFrame:
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


def montar_linha(data: str, horarios: List[str]) -> dict:
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


def completar_ano_em_ddmm(texto_data: str, ano_ref: str) -> str:
    texto_data = str(texto_data).strip()
    if re.fullmatch(r"\d{2}/\d{2}", texto_data) and ano_ref:
        return f"{texto_data}/{ano_ref}"
    return texto_data


def extrair_ano_referencia(texto: str) -> str:
    anos = re.findall(r"\b(20\d{2})\b", texto)
    if anos:
        return anos[0]
    return ""


def extrair_mes_ano_periodo(texto: str) -> Tuple[str, str]:
    # tenta pegar um período do tipo 21/07/2024 a 20/08/2024
    m = re.search(r"(\d{2})/(\d{2})/(\d{4})\s*a\s*(\d{2})/(\d{2})/(\d{4})", texto)
    if m:
        return m.group(2), m.group(3)
    return "", ""


# =========================================================
# PRÉ-PROCESSAMENTO DE IMAGEM
# =========================================================
def corrigir_rotacao(pil_img: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(pil_img)
        m = re.search(r"Rotate: (\d+)", osd)
        if m:
            angle = int(m.group(1))
            if angle == 90:
                return pil_img.rotate(-90, expand=True)
            elif angle == 180:
                return pil_img.rotate(180, expand=True)
            elif angle == 270:
                return pil_img.rotate(90, expand=True)
    except Exception:
        pass
    return pil_img


def cortar_metade_direita(pil_img: Image.Image) -> Image.Image:
    largura, altura = pil_img.size
    return pil_img.crop((largura // 2, 0, largura, altura))


def preprocess_image(pil_img: Image.Image) -> Image.Image:
    img = np.array(pil_img)

    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    return Image.fromarray(gray)


def ocr_image(img: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(img, lang="por", config="--psm 6")
    except Exception:
        return pytesseract.image_to_string(img, config="--psm 6")


def extract_text(file_bytes: bytes, file_type: str) -> str:
    partes = []

    if file_type == "application/pdf":
        paginas = convert_from_bytes(file_bytes, dpi=150)
        for pagina in paginas:
            pagina = corrigir_rotacao(pagina)

            # OCR preliminar para detectar contracheque + cartão
            texto_pre = ocr_image(preprocess_image(pagina))
            if "CARTÃO DE PONTO" in texto_pre.upper() and (
                "DEMONSTRATIVO DE PAGAMENTO" in texto_pre.upper() or
                "SALÁRIO" in texto_pre.upper()
            ):
                pagina = cortar_metade_direita(pagina)

            processada = preprocess_image(pagina)
            partes.append(ocr_image(processada))
    else:
        imagem = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        imagem = corrigir_rotacao(imagem)

        texto_pre = ocr_image(preprocess_image(imagem))
        if "CARTÃO DE PONTO" in texto_pre.upper() and (
            "DEMONSTRATIVO DE PAGAMENTO" in texto_pre.upper() or
            "SALÁRIO" in texto_pre.upper()
        ):
            imagem = cortar_metade_direita(imagem)

        processada = preprocess_image(imagem)
        partes.append(ocr_image(processada))

    return "\n".join(partes)


# =========================================================
# PARSERS DOS MODELOS
# =========================================================
def parse_modelo_1(texto: str) -> pd.DataFrame:
    # Modelo com coluna "Marcações", horários simples na mesma linha da data
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    padrao_data = re.compile(r"\d{2}/\d{2}/\d{4}")
    padrao_hora = re.compile(r"\b\d{2}:\d{2}\b")

    registros = []
    for linha in linhas:
        data_match = padrao_data.search(linha)
        if data_match:
            data = data_match.group()
            horarios = padrao_hora.findall(linha)
            registros.append(montar_linha(data, horarios))

    if not registros:
        return pd.DataFrame(columns=COLUNAS_MODELO)

    return preparar_df(pd.DataFrame(registros))


def parse_modelo_2(texto: str) -> pd.DataFrame:
    # Modelo com "Marcações" e pares com hífen
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    padrao_data = re.compile(r"\b\d{2}\b")
    padrao_hora = re.compile(r"\b\d{2}:\d{2}\b")

    registros = []
    for linha in linhas:
        if not padrao_data.search(linha):
            continue

        horas = padrao_hora.findall(linha)
        if horas:
            registros.append(montar_linha("", horas))
        else:
            registros.append(montar_linha("", []))

    if not registros:
        return pd.DataFrame(columns=COLUNAS_MODELO)

    return preparar_df(pd.DataFrame(registros))


def parse_modelo_3(texto: str) -> pd.DataFrame:
    # Modelo tabular com Ent.1 / Sai.1 / Ent.2 / Sai.2 ...
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    padrao_data = re.compile(r"\b\d{2}/\d{2}\b")
    padrao_hora = re.compile(r"\b\d{2}:\d{2}\b")

    ano_ref = extrair_ano_referencia(texto)
    registros = []

    for linha in linhas:
        data_match = padrao_data.search(linha)
        if not data_match:
            continue

        data = completar_ano_em_ddmm(data_match.group(), ano_ref)
        horarios = padrao_hora.findall(linha)
        registros.append(montar_linha(data, horarios))

    if not registros:
        return pd.DataFrame(columns=COLUNAS_MODELO)

    return preparar_df(pd.DataFrame(registros))


def parse_modelo_4(texto: str) -> pd.DataFrame:
    # Modelo textual com ENTRA / I.INI / I.FIN / SAIDA
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    padrao_data = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
    padrao_hora = re.compile(r"\b\d{2}:\d{2}\b")

    registros = []
    for linha in linhas:
        data_match = padrao_data.search(linha)
        if not data_match:
            continue

        data = data_match.group()
        horarios = padrao_hora.findall(linha)
        registros.append(montar_linha(data, horarios))

    if not registros:
        return pd.DataFrame(columns=COLUNAS_MODELO)

    return preparar_df(pd.DataFrame(registros))


def parse_modelo_5(texto: str) -> pd.DataFrame:
    # Modelo de cartão ao lado de contracheque; dia do mês + horários
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    padrao_dia = re.compile(r"^\d{2}$")
    padrao_hora = re.compile(r"\b\d{2}:\d{2}\b")

    mes_ref, ano_ref = extrair_mes_ano_periodo(texto)
    registros = []

    for linha in linhas:
        partes = linha.split()
        if not partes:
            continue

        if padrao_dia.match(partes[0]):
            dia = partes[0]
            data = f"{dia}/{mes_ref}/{ano_ref}" if mes_ref and ano_ref else dia
            horarios = padrao_hora.findall(linha)
            registros.append(montar_linha(data, horarios))

    if not registros:
        return pd.DataFrame(columns=COLUNAS_MODELO)

    return preparar_df(pd.DataFrame(registros))


def detectar_modelo(texto: str) -> str:
    t = texto.upper()

    if "ENTRA" in t and "I.INI" in t and "I.FIN" in t and "SAIDA" in t:
        return "modelo_4"

    if "ENT.1" in t and "SAI.1" in t:
        return "modelo_3"

    if "CARTÃO DE PONTO" in t and "MARCAÇÕES" in t and "TRAB" in t:
        return "modelo_5"

    if "MARCAÇÕES" in t and "-" in t:
        return "modelo_2"

    if "MARCAÇÕES" in t:
        return "modelo_1"

    # fallback genérico por data completa
    if re.search(r"\b\d{2}/\d{2}/\d{4}\b", texto):
        return "modelo_4"

    return "modelo_1"


def parse_cartao(texto: str) -> pd.DataFrame:
    modelo = detectar_modelo(texto)

    if modelo == "modelo_4":
        return parse_modelo_4(texto)
    if modelo == "modelo_3":
        return parse_modelo_3(texto)
    if modelo == "modelo_5":
        return parse_modelo_5(texto)
    if modelo == "modelo_2":
        return parse_modelo_2(texto)
    return parse_modelo_1(texto)


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
    '<div class="muted">A grade abaixo permite adicionar e excluir linhas diretamente nela.</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

if arquivo is None:
    st.info("Envie um arquivo para iniciar.")
    st.stop()

# =========================================================
# PROCESSAMENTO DO ARQUIVO
# =========================================================
try:
    file_bytes = arquivo.read()
    file_hash = hash_arquivo(file_bytes)
    arquivo_novo = st.session_state.arquivo_hash != file_hash

    if arquivo_novo:
        with st.spinner("Lendo e organizando o cartão..."):
            texto = extract_text(file_bytes, arquivo.type)
            df_lido = parse_cartao(texto)
            df_lido = preparar_df(df_lido)

        st.session_state.arquivo_hash = file_hash
        st.session_state.ocr_bruto = texto
        st.session_state.df_original = df_lido.copy()
        st.session_state.df_editado = df_lido.copy()

except Exception as e:
    st.error("Erro ao ler o arquivo.")
    st.exception(e)
    st.stop()

# =========================================================
# AÇÕES
# =========================================================
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🔄 Reprocessar arquivo", use_container_width=True):
        try:
            with st.spinner("Relendo o cartão..."):
                texto = extract_text(file_bytes, arquivo.type)
                df_lido = parse_cartao(texto)
                df_lido = preparar_df(df_lido)

            st.session_state.ocr_bruto = texto
            st.session_state.df_original = df_lido.copy()
            st.session_state.df_editado = df_lido.copy()
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

with st.expander("Visualizar OCR bruto"):
    st.text(st.session_state.ocr_bruto[:5000] if st.session_state.ocr_bruto else "")

# =========================================================
# TABELA PRINCIPAL
# =========================================================
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

# =========================================================
# RELATÓRIOS DINÂMICOS
# =========================================================
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
