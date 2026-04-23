from fastapi import FastAPI, UploadFile
import pdfplumber
import pytesseract
from PIL import Image
import io, os, json
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ocr_imagem(img_bytes):
    imagem = Image.open(io.BytesIO(img_bytes))
    imagem = imagem.convert("L")  # melhora OCR
    return pytesseract.image_to_string(imagem, lang="por")

def extrair_texto_pdf(file_bytes):
    texto = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texto += t + "\n"

            # se não houver texto, faz OCR da página
            if not t:
                imagem = page.to_image(resolution=300)
                texto += ocr_imagem(imagem.original.tobytes())

    return texto

def extrair_texto(file_bytes, filename):
    if filename.lower().endswith(".pdf"):
        return extrair_texto_pdf(file_bytes)
    else:
        return ocr_imagem(file_bytes)

@app.post("/ler-cartao")
async def ler_cartao(file: UploadFile):
    conteudo = await file.read()
    texto = extrair_texto(conteudo, file.filename)

    prompt = f"""
Você está lendo um CARTÃO DE PONTO brasileiro.

O texto pode estar bagunçado, fora de ordem, com erros de OCR.

Sua tarefa é encontrar TODAS as datas e TODAS as batidas de horário.

Responda SOMENTE neste JSON, exatamente neste formato:

[
  {{
    "Data": "dd/mm/aaaa",
    "Entrada1": "",
    "Saida1": "",
    "Entrada2": "",
    "Saida2": ""
  }}
]

Regras:
- Cada linha é um dia.
- Até 4 batidas.
- Se faltar horário, deixe "".
- Ignore textos irrelevantes.

Texto extraído:
{texto}
"""

    resposta = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )

    conteudo = resposta.output[0].content[0].text
    return json.loads(conteudo)
