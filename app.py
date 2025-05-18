import os
import base64
import json
import re
import cv2
import tempfile
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import numpy as np
from PIL import Image

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def main():
    st.title("Laudo Técnico feito por IA")

    uploaded_files = st.file_uploader(
        "Envie imagens para análise",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.warning("Envie ao menos uma imagem para gerar os laudos.")
        return

    if st.button("Gerar Laudo"):
        with st.spinner("Processando imagens... Isso pode levar um tempo"):
            imagens_bytes = [(f.name, f.read()) for f in uploaded_files]
            imagens_processadas = gerar_imagem_laudo(imagens_bytes)
            laudos_textuais = gerar_texto_laudo(imagens_bytes)

            st.success("Laudos gerados com sucesso!")
            gerar_pdf(laudos_textuais, imagens_processadas)

def gerar_texto_laudo(imagens_bytes):
    laudos_textuais = []

    for nome, image_bytes in imagens_bytes:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        prompt = """
        Com base na imagem fornecida, elabore um laudo técnico de inspeção predial, seguindo a estrutura abaixo:

        Elemento Inspecionado: Identifique qual parte da edificação está representada na imagem (ex: fachada, laje, revestimento interno, cobertura, etc.).

        Localização: Informe a localização aproximada do problema (ex: bloco A, fachada lateral esquerda, 3º pavimento, etc.).

        Descrição da Patologia: Descreva detalhadamente o problema observado na imagem (ex: trincas verticais, eflorescência, infiltrações, descolamento de revestimento, etc.).

        Causa Provável: Aponte a(s) possível(is) causa(s) da patologia (ex: má execução, ausência de impermeabilização, movimentação térmica, etc.).

        Recomendações Técnicas: Indique as ações corretivas recomendadas para o problema identificado (ex: recuperação estrutural, substituição de revestimento, impermeabilização, etc.).

        Utilize linguagem técnica, objetiva e formal, conforme o padrão de laudos de inspeção predial, e evite suposições vagas.
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0
        )

        texto_laudo = response.choices[0].message.content
        laudos_textuais.append(texto_laudo)

    return laudos_textuais

def gerar_imagem_laudo(imagens_bytes):
    imagens_processadas = []

    for nome, image_bytes in imagens_bytes:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        np_arr = np.frombuffer(image_bytes, np.uint8)
        imagem_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if imagem_cv is None:
            st.warning(f"Erro ao carregar imagem: {nome}")
            continue

        altura, largura = imagem_cv.shape[:2]

        prompt = f"""
        Você irá analisar uma imagem de inspeção predial.

        A imagem tem largura {largura} pixels e altura {altura} pixels.

        Identifique todos os problemas visíveis e para cada problema retorne no JSON:

        - descricao: texto curto do problema (ex: 'infiltração', 'rachadura', etc.)
        - ponto: [x, y] coordenadas inteiras, indicando o ponto mais representativo do problema
        - bounding_box: [xmin, ymin, xmax, ymax] delimitador aproximado

        Se não encontrar problemas, retorne "problemas": []
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0
        )

        resposta_texto = response.choices[0].message.content.strip()
        resposta_texto = re.sub(r"^```json\s*", "", resposta_texto)
        resposta_texto = re.sub(r"\s*```$", "", resposta_texto)

        try:
            dados = json.loads(resposta_texto)
            problemas = dados.get("problemas", [])
        except:
            st.warning(f"Erro ao interpretar JSON para a imagem {nome}")
            continue

        for problema in problemas:
            descricao = problema.get("descricao", "problema")
            ponto = problema.get("ponto", [0, 0])
            bounding_box = problema.get("bounding_box", [0, 0, 0, 0])

            x, y = ponto
            xmin, ymin, xmax, ymax = bounding_box

            # Garantir coordenadas válidas
            xmin = max(0, min(xmin, largura - 1))
            xmax = max(0, min(xmax, largura - 1))
            ymin = max(0, min(ymin, altura - 1))
            ymax = max(0, min(ymax, altura - 1))
            x = max(0, min(x, largura - 1))
            y = max(0, min(y, altura - 1))

            cv2.rectangle(imagem_cv, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
            cv2.circle(imagem_cv, (x, y), 12, (0, 0, 255), -1)
            cv2.putText(imagem_cv, descricao, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        imagens_processadas.append(imagem_cv)

    return imagens_processadas

def gerar_pdf(laudos_textuais, imagens_processadas):
    if not laudos_textuais or not imagens_processadas:
        st.error("Erro ao gerar PDF: lista de imagens ou laudos está vazia.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        doc = SimpleDocTemplate(tmpfile.name, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        style_normal = styles["Normal"]

        for imagem_cv, laudo in zip(imagens_processadas, laudos_textuais):
            imagem_rgb = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(imagem_rgb)

            img_buffer = BytesIO()
            pil_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            # Calcula tamanho proporcional
            largura_original, altura_original = pil_img.size
            nova_largura = 400
            nova_altura = int((nova_largura / largura_original) * altura_original)

            rl_image = RLImage(img_buffer, width=nova_largura, height=nova_altura)
            elements.append(rl_image)
            elements.append(Spacer(1, 12))

            for linha in laudo.strip().split("\n"):
                elements.append(Paragraph(linha.strip(), style_normal))
            elements.append(Spacer(1, 24))

        doc.build(elements)

        with open(tmpfile.name, "rb") as f:
            pdf_bytes = f.read()

        st.download_button("Baixar PDF com Laudos", data=pdf_bytes,
                           file_name="laudo_tecnico.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
