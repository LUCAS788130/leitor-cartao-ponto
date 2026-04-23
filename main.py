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

# =========================
# ESTILO
# =========================
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

.kpi {
    background: linear-gradient(135deg, #111827, #0f172a);
    border: 1px solid #243041;
    border-radius: 16px;
    padding: 14px 16px;
    min-height: 88px;
}

.kpi-titulo {
    color: #94a3b8;
    font-size: 0.92rem;
    margin-bottom: 6px;
}

.kpi-valor {
    font-size: 1.6rem;
    font-weight: 700;
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

# =========================
# FUNÇÕES
# =========================
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

def excluir_linhas_por_indice(df, indices):
    if not indices:
        return df.copy()
    return df.drop(index=indices, errors="ignore").reset_index(drop=True)

def validar_pares(df):
    df = df.copy()
    inconsistencias = []
    horarios_sem_par = []

    for _, row in df.iterrows():
        horarios_problematicos = []

        for i in range(1, 7):
            entrada_col = f"Entrada{i}"
            saida_col = f"Saída{i}"

            entrada = str(row.get(entrada_col, "")).strip()
            saida = str(row.get(saida_col, "")).strip()

            if entrada and not saida:
                horarios_problematicos.append(f"{entrada_col}: {entrada}")

            if saida and not entrada:
                horarios_problematicos.append(f"{saida_col}: {saida}")

        inconsistencias.append("Sim" if horarios_problematicos else "")
        horarios_sem_par.append(" | ".join(horarios_problematicos))

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

def obter_hash_arquivo(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def preparar_df_exportacao(df):
    df = df.copy()
    for col in COLUNAS_MODELO:
        if col not in df.columns:
            df[col] = ""
    return df[COLUNAS_MODELO]

# =========================
# ESTADO
# =========================
if "arquivo_hash" not in st.session_state:
    st.session_state.arquivo_hash = None

if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

if "df_base" not in st.session_state:
    st.session_state.df_base = pd.DataFrame(columns=COLUNAS_MODELO)

if "df_editado" not in st.session_state:
    st.session_state.df_editado = pd.DataFrame(columns=COLUNAS_MODELO)

if "linhas_excluir" not in st.session_state:
    st.session_state.linhas_excluir = []

# =========================
# UPLOAD
# =========================
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
        st.session_state.df_base = df_lido.copy()
        st.session_state.df_editado = df_lido.copy()
        st.session_state.linhas_excluir = []

    # KPIs
    df_validado_kpi = validar_pares(st.session_state.df_editado)
    df_incons_kpi = gerar_relatorio_inconsistencias(st.session_state.df_editado)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-titulo">Total de linhas</div>
            <div class="kpi-valor">{len(df_validado_kpi)}</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        total_datas = int((df_validado_kpi["Data"].astype(str).str.strip() != "").sum()) if not df_validado_kpi.empty else 0
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-titulo">Dias identificados</div>
            <div class="kpi-valor">{total_datas}</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-titulo">Horários sem par</div>
            <div class="kpi-valor">{len(df_incons_kpi)}</div>
        </div>
        """, unsafe_allow_html=True)

    # Ações
    a1, a2, a3, a4 = st.columns(4)

    with a1:
        if st.button("➕ Adicionar linha", use_container_width=True):
            st.session_state.df_editado = adicionar_linha_vazia(st.session_state.df_editado)
            st.rerun()

    with a2:
        if st.button("🔄 Reprocessar arquivo", use_container_width=True):
            with st.spinner("Relendo o cartão..."):
                raw_text = extract_text(file_bytes, uploaded_file.type)
                df_lido = parse_cartao(raw_text)

            st.session_state.raw_text = raw_text
            st.session_state.df_base = df_lido.copy()
            st.session_state.df_editado = df_lido.copy()
            st.session_state.linhas_excluir = []
            st.rerun()

    with a3:
        if st.button("🗑️ Excluir linhas marcadas", use_container_width=True):
            st.session_state.df_editado = excluir_linhas_por_indice(
                st.session_state.df_editado,
                st.session_state.linhas_excluir
            )
            st.session_state.linhas_excluir = []
            st.rerun()

    with a4:
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

    df_exibicao = validar_pares(st.session_state.df_editado)

    selecao = st.multiselect(
        "Marque os números das linhas que deseja excluir",
        options=list(range(len(df_exibicao))),
        default=st.session_state.linhas_excluir,
        format_func=lambda x: f"Linha {x + 1} - {df_exibicao.iloc[x]['Data'] if x < len(df_exibicao) else ''}"
    )
    st.session_state.linhas_excluir = selecao

    edited_df = st.data_editor(
        df_exibicao,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="editor_cartao_principal",
        disabled=["Inconsistência", "Horários sem par"],
        column_config={
            "Data": st.column_config.TextColumn("Data", width="medium"),
            "Entrada1": st.column_config.TextColumn("Entrada1"),
            "Saída1": st.column_config.TextColumn("Saída1"),
            "Entrada2": st.column_config.TextColumn("Entrada2"),
            "Saída2": st.column_config.TextColumn("Saída2"),
            "Entrada3": st.column_config.TextColumn("Entrada3"),
            "Saída3": st.column_config.TextColumn("Saída3"),
            "Entrada4": st.column_config.TextColumn("Entrada4"),
            "Saída4": st.column_config.TextColumn("Saída4"),
            "Entrada5": st.column_config.TextColumn("Entrada5"),
            "Saída5": st.column_config.TextColumn("Saída5"),
            "Entrada6": st.column_config.TextColumn("Entrada6"),
            "Saída6": st.column_config.TextColumn("Saída6"),
            "Inconsistência": st.column_config.TextColumn("Inconsistência", width="small"),
            "Horários sem par": st.column_config.TextColumn("Horários sem par", width="large"),
        }
    )

    st.session_state.df_editado = preparar_df_exportacao(pd.DataFrame(edited_df))

    st.subheader("Validação de horários")
    df_inconsistencias = gerar_relatorio_inconsistencias(st.session_state.df_editado)

    if len(df_inconsistencias) > 0:
        st.error(f"Foram encontrados {len(df_inconsistencias)} horário(s) sem par.")
        st.dataframe(df_inconsistencias, use_container_width=True, hide_index=True)
    else:
        st.success("Nenhum horário sem par encontrado.")

else:
    st.info("Envie um arquivo para iniciar a leitura do cartão.")
