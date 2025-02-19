import os
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Título de la aplicación
st.title('PREDICCIÓN ELECTORAL XLSX')

# Inicializar mensajes en sesión
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos en el chat
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

def clean_text(text):
    """Limpia el texto eliminando caracteres especiales y normalizándolo."""
    text = re.sub(r'\s+', ' ', text)  # Elimina espacios extra
    text = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\s]', '', text)  # Elimina caracteres especiales
    return text.strip()

def extract_text_from_excel(excel_file):
    """Lee un archivo Excel y convierte su contenido en texto limpio."""
    df = pd.read_excel(excel_file, sheet_name=None)  # Leer todas las hojas
    extracted_text = ""
    for sheet_name, sheet_data in df.items():
        extracted_text += f"--- {sheet_name} ---\n"
        extracted_text += sheet_data.to_string(index=False) + "\n\n"
    return clean_text(extracted_text[:4000])  # Limitar a 4000 caracteres

# Subir archivo Excel
uploaded_file = st.file_uploader('Sube un archivo Excel', type=['xlsx', 'xls'])

if uploaded_file:
    with st.chat_message('user'):
        st.markdown(f"Analizando:**{uploaded_file.name}**")

    # Extraer y limpiar datos del archivo
    extracted_text = extract_text_from_excel(uploaded_file)

    # Mostrar el contenido del archivo extraído para depuración
    st.text_area("Datos extraídos (Muestra):", extracted_text[:500], height=200)

    # Guardar mensaje en el historial de la sesión
    st.session_state.messages.append({'role': 'user', 'content': f'Subido: {uploaded_file.name}'})

    with st.chat_message('assistant'):
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": 
                 "Analiza los datos y clasifica las menciones de votos en: 'VOTO NOBOA', 'VOTO LUISA', 'VOTO NULO'. "
                 "Muestra un conteo de cada categoría y genera un gráfico de barras con los resultados."},
                {"role": "user", "content": extracted_text}
            ]
        )
        
        assistant_reply = response.choices[0].message.content
        st.markdown(assistant_reply)

    st.session_state.messages.append({'role': 'assistant', 'content': assistant_reply})

    # Contar menciones de votos en base a las conclusiones de la IA
    vote_counts = {
        "VOTO NOBOA": assistant_reply.lower().count("voto noboa"),
        "VOTO LUISA": assistant_reply.lower().count("voto luisa"),
        "VOTO NULO": assistant_reply.lower().count("voto nulo")
    }
    
    df_votes = pd.DataFrame(list(vote_counts.items()), columns=["Categoría", "Cantidad"])

    # Generar gráfico dinámico según el análisis de la IA
    fig, ax = plt.subplots()
    ax.bar(df_votes["Categoría"], df_votes["Cantidad"], color=['blue', 'green', 'red'])
    ax.set_ylabel("Cantidad de Votos")
    ax.set_title("Distribución de Votos")
    st.pyplot(fig)

    # Sección para hacer preguntas sobre los datos cargados
    user_question = st.text_input("❓ Pregunta sobre los datos cargados:")
    
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": 
                 "Responde preguntas basadas en los datos de votos cargados."},
                {"role": "user", "content": f"Datos cargados: {extracted_text}"},
                {"role": "user", "content": user_question}
            ]
        )

        answer = response.choices[0].message.content

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({'role': 'assistant', 'content': answer})
