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

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

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
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 96%;
}

h1, h2, h3 {
    letter-spacing: -0.3px;
}

.card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 18px 18px 14px 18px;
    margin-bottom: 14px;
}

.card-light {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 14px;
}

.small-muted {
    color: #94a3b8;
    font-size: 0.92rem;
}

.kpi {
    background: linear-gradient(135deg, #0f172a, #111827);
    border: 1px solid #243041;
    border-radius: 16px;
    padding: 14px 16px;
    min-height: 90px;
}

.kpi-title {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 6px;
}

.kpi-value {
    font-size: 1.7rem;
    font-weight: 700;
}

hr {
    border-color: #1f2937;
}
</style>
""", unsafe_allow_html=True)

st.title("🕒 Leitor de Cartão de Ponto")
st.caption("Converte cartões de ponto para o modelo exato de importação, com revisão manual antes da exportação.")

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

        inconsistencias.append("Sim" if problemas else "")
        horarios_sem_par.append(" | ".join(horarios_problematicos) if horarios_problematicos else "")

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

def preparar_df_para_exportacao(df):
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

# =========================
# UPLOAD
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Envie o cartão de ponto",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Aceita PDF, JPG, JPEG e PNG."
)
st.markdown('<div class="small-muted">O sistema lê o cartão, organiza os dias e horários no modelo exato e permite revisão antes do CSV.</div>', unsafe_allow_html=True)
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

    # =========================
    # AÇÕES
    # =========================
    col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])

    with col_a:
        if st.button("➕ Adicionar linha", use_container_width=True):
            st.session_state.df_editado = adicionar_linha_vazia(st.session_state.df_editado)

    with col_b:
        if st.button("🔄 Reprocessar arquivo", use_container_width=True):
            with st.spinner("Relendo o cartão..."):
                raw_text = extract_text(file_bytes, uploaded_file.type)
                df_lido = parse_cartao(raw_text)

            st.session_state.raw_text = raw_text
            st.session_state.df_base = df_lido.copy()
            st.session_state.df_editado = df_lido.copy()
            st.rerun()

    with col_c:
        if st.button("🧹 Limpar tabela", use_container_width=True):
            st.session_state.df_editado = pd.DataFrame(columns=COLUNAS_MODELO)
            st.rerun()

    with col_d:
        csv_data = preparar_df_para_exportacao(st.session_state.df_editado).to_csv(
            index=False, encoding="utf-8-sig"
        ).encode("utf-8-sig")

        st.download_button(
            "📥 Baixar CSV",
            data=csv_data,
            file_name="cartao_ponto_modelo_exato.csv",
            mime="text/csv",
            use_container_width=True
        )

    # =========================
    # KPIS
    # =========================
    df_validado = validar_pares(st.session_state.df_editado)
    df_inconsistencias = gerar_relatorio_inconsistencias(st.session_state.df_editado)

    total_linhas = len(df_validado)
    total_incons = len(df_inconsistencias)
    total_com_data = int((df_validado["Data"].astype(str).str.strip() != "").sum()) if not df_validado.empty else 0

    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-title">Dias na tabela</div>
            <div class="kpi-value">{total_linhas}</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-title">Dias com data preenchida</div>
            <div class="kpi-value">{total_com_data}</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-title">Horários sem par</div>
            <div class="kpi-value">{total_incons}</div>
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # OCR BRUTO
    # =========================
    with st.expander("Visualizar OCR bruto"):
        st.text(st.session_state.raw_text[:5000] if st.session_state.raw_text else "")

    # =========================
    # GRADE EDITÁVEL
    # =========================
    st.subheader("Tabela final para revisão")

    df_exibicao = validar_pares(st.session_state.df_editado)

    gb = GridOptionsBuilder.from_dataframe(df_exibicao)
    gb.configure_default_column(editable=True, resizable=True, filter=True, sortable=True)

    gb.configure_selection(selection_mode="multiple", use_checkbox=True)

    gb.configure_column("Inconsistência", editable=False)
    gb.configure_column("Horários sem par", editable=False)
    gb.configure_column("Data", minWidth=120)

    for col in COLUNAS_MODELO[1:]:
        gb.configure_column(col, minWidth=95)

    gb.configure_grid_options(
        rowHeight=34,
        headerHeight=38,
        animateRows=False
    )

    grid_response = AgGrid(
        df_exibicao,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
        height=420,
        allow_unsafe_jscode=False,
        theme="streamlit"
    )

    df_editado_novo = pd.DataFrame(grid_response["data"])
    st.session_state.df_editado = preparar_df_para_exportacao(df_editado_novo)

    selecionadas = grid_response.get("selected_rows", [])
    qtd_selecionadas = len(selecionadas) if selecionadas is not None else 0

    col_x, col_y = st.columns([1, 4])

    with col_x:
        if st.button("🗑️ Excluir selecionadas", use_container_width=True, disabled=(qtd_selecionadas == 0)):
            df_atual = pd.DataFrame(grid_response["data"]).copy()

            if qtd_selecionadas > 0:
                df_sel = pd.DataFrame(selecionadas).copy()

                # garante comparação pelas colunas do modelo
                cols_compare = [c for c in COLUNAS_MODELO if c in df_atual.columns and c in df_sel.columns]
                if cols_compare:
                    for c in cols_compare:
                        df_atual[c] = df_atual[c].fillna("").astype(str)
                        df_sel[c] = df_sel[c].fillna("").astype(str)

                    idx_drop = []
                    for _, row_sel in df_sel.iterrows():
                        mask = pd.Series(True, index=df_atual.index)
                        for c in cols_compare:
                            mask &= (df_atual[c] == row_sel[c])
                        encontrados = df_atual[mask]
                        if not encontrados.empty:
                            idx_drop.append(encontrados.index[0])

                    df_atual = df_atual.drop(index=idx_drop).reset_index(drop=True)

                st.session_state.df_editado = preparar_df_para_exportacao(df_atual)
                st.rerun()

    with col_y:
        st.caption("Selecione uma ou mais linhas na caixa de seleção à esquerda para excluí-las.")

    # =========================
    # INCONSISTÊNCIAS
    # =========================
    st.subheader("Validação de horários")

    df_inconsistencias = gerar_relatorio_inconsistencias(st.session_state.df_editado)

    if len(df_inconsistencias) > 0:
        st.error(f"Foram encontrados {len(df_inconsistencias)} horário(s) sem par.")

        st.dataframe(
            df_inconsistencias,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("Nenhum horário sem par encontrado.")

else:
    st.info("Envie um arquivo para iniciar a leitura do cartão.")
