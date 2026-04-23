from fastapi import FastAPI, UploadFile
import pdfplumber, pytesseract
from PIL import Image
import io, os, json
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extrair_texto(file_bytes, filename):
    if filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            texto = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if texto.strip():
                return texto

    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image, lang="por")

@app.post("/ler-cartao")
async def ler_cartao(file: UploadFile):
    conteudo = await file.read()
    texto = extrair_texto(conteudo, file.filename)

    prompt = f"""
Você está lendo um CARTÃO DE PONTO.

Extraia TODAS as datas e TODAS as batidas de horário.

Responda OBRIGATORIAMENTE neste JSON, exatamente neste formato:

[
  {{
    "Data": "dd/mm/aaaa",
    "Entrada1": "hh:mm",
    "Saida1": "hh:mm",
    "Entrada2": "hh:mm",
    "Saida2": "hh:mm"
  }}
]

Se faltar batida, deixe vazio "".

Texto do cartão:
{texto}
"""

    resposta = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )

    conteudo = resposta.output[0].content[0].text
    return json.loads(conteudo)
