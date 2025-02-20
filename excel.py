import os
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from groq import Groq
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
logger.info(os.getenv('GROQ_API_KEY'))

qclient = Groq()

# Título de la aplicación
st.title('PREDICCIÓN ELECTORAL XLSX')

# Inicializar mensajes en sesión
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Función para limpiar texto
def clean_text(text):     
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\s]', '', text) 
    return text.strip()

# Función para extraer texto de un archivo Excel
def extract_text_from_excel(excel_file):
    df = pd.read_excel(excel_file, sheet_name=None)  
    text_fragments = []
    
    for sheet_name, sheet_data in df.items():
        text = f"--- {sheet_name} ---\n" + sheet_data.to_string(index=False)
        cleaned_text = clean_text(text)
        
        fragment_size = 1000
        fragments = [cleaned_text[i:i+fragment_size] for i in range(0, len(cleaned_text), fragment_size)]
        text_fragments.extend(fragments)
    
    return text_fragments

uploaded_file = st.file_uploader('Sube un archivo Excel', type=['xlsx', 'xls'])

if uploaded_file:
    with st.chat_message('user'):
        st.markdown(f"Analizando: **{uploaded_file.name}**")

    extracted_text_fragments = extract_text_from_excel(uploaded_file)

    st.text_area("Datos extraídos (Muestra):", extracted_text_fragments[0][:500], height=200)

    st.session_state.messages.append({'role': 'user', 'content': f'Subido: {uploaded_file.name}'})

    vote_counts = {"VOTO NOBOA": 150, "VOTO LUISA": 120, "VOTO NULO": 30}

    # Cargar el modelo de Hugging Face para clasificación o análisis de texto
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

    for text_fragments in extracted_text_fragments:
        # Clasificar el fragmento
        result = classifier(text_fragments)
        # Suponiendo que el resultado contiene categorías de votos
        if "VOTO NOBOA" in result[0]['label']:
            vote_counts["VOTO NOBOA"] += 1
        elif "VOTO LUISA" in result[0]['label']:
            vote_counts["VOTO LUISA"] += 1
        elif "VOTO NULO" in result[0]['label']:
            vote_counts["VOTO NULO"] += 1

    df_votes = pd.DataFrame(list(vote_counts.items()), columns=["Categoría", "Cantidad"])

    # Generar gráfico de barras con los resultados
    fig, ax = plt.subplots()
    ax.bar(df_votes["Categoría"], df_votes["Cantidad"], color=['blue', 'green', 'red'])
    ax.set_ylabel("Cantidad de Votos")
    ax.set_title("Distribución de Votos")
    st.pyplot(fig)


    # Sección para hacer preguntas sobre los datos cargados
    user_question = st.text_input("Pregunta sobre los datos cargados:")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)

        # Responder con el modelo
        response = classifier(user_question)
        answer = response[0]['label']

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({'role': 'assistant', 'content': answer})


    with st.chat_message('assistant'):
        stream_response = qclient.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": " VAS A ANALIZAR EL TEXTO Y SI HAY INTENCION DE CORREO O LUISA, SUMA UN VOTO A LUISA, SI TEIEN REFERENCIA A NOBOA SUMA UN VOTO A NOBOA, SI NO TIENE REFERENCIA SUMA UN BOTO A NULO ",
                },
                {
                    "role": "user",
                    "content": extracted_text_fragments,
                },
            ],
            model="llama3-8b-8192",
            stream=True
        )

        response = extracted_text_fragments(stream_response)
        response = st.write_stream(response)

    st.session_state.messages.append({'role': 'assistant', 'content': response})









































