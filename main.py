from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

import openai
from streamlit_chat import message
from utils import *


from dotenv import load_dotenv
import os


import streamlit as st

# Define las variables de entorno y otros secrets
st.markdown("""
    <style>
        .stGrid .stGridCell {
            padding: 0px;
        }
    </style>
    """, unsafe_allow_html=True)

st.header("Consulta lo que quieras sobre los pliegos RIC 💬 📘")

# Mostrando el título y las imágenes
# Creación de columnas para la imagen y el texto. 
# Haremos las proporciones iguales para que ocupen todo el ancho.
col1, col2 = st.columns([1, 5]) 

# En la primera columna, mostramos la imagen
with col1:
    st.markdown(
        """
        <img src="https://img.freepik.com/vector-premium/rayo-plano-simple-ilustracion-energia-electrica-simbolo-energia-electricidad_606097-132.jpg" width="100%" align="left">
        """,
        unsafe_allow_html=True,
    )

# En la segunda columna, mostramos el título
with col2:
    st.markdown("<h6 style='text-align: left;'>Creado por: Robinson Cornejo, Instalador Electrico Clase A </h6>", unsafe_allow_html=True)
    st.markdown(
    """
    <h6 style='text-align: left;'>
        <a href="https://www.linkedin.com/in/robinsonfce/" target="_blank">LinkedIn</a> | 
        <a href="mailto:robinson.fce@gmail.com">Correo</a>
    </h6>
    """, 
    unsafe_allow_html=True
)




# Inicialización de mensajes
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "En qué puedo ayudarte, explicame tu pregunta con detalle"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    # Carga el modelo y la información. Aquí deberías adaptar cómo cargas tu modelo/data.
    # Como es solo un ejemplo, he mantenido la estructura original.
    with st.spinner(text="Cargando información necesaria..."):
        # Aquí deberías tener tu inicialización de modelo y datos, similar a lo que tenías antes.
        return "Model/Data Loaded"  # Solo un marcador de posición

load_data()

chat_engine = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

# Aceptar entrada del usuario
if prompt := st.chat_input("Tu consulta"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Mostrar los mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Si el último mensaje no es del asistente, genera una nueva respuesta
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            conversation_string = "Tu conversación actual o histórico aquí"  # Ajusta según sea necesario
            refined_query = query_refiner(conversation_string, prompt)
            response = get_answer(refined_query)
            response_content = response["output_text"]
            st.write(response_content)
            message = {"role": "assistant", "content": response_content}
            st.session_state.messages.append(message)