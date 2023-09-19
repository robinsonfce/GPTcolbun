from langchain.chat_models import ChatOpenAI
import openai
from streamlit_chat import message
from utils import *
from dotenv import load_dotenv
import os
import streamlit as st

# Estilos y definiciones iniciales
st.markdown("""
    <style>
        .stGrid .stGridCell {
            padding: 0px;
        }
    </style>
    """, unsafe_allow_html=True)

st.header("Consulta lo que quieras sobre los pliegos RIC 💬 📘")

col1, col2 = st.columns([1, 5])

with col1:
    st.markdown(
        """
        <img src="https://img.freepik.com/vector-premium/rayo-plano-simple-ilustracion-energia-electrica-simbolo-energia-electricidad_606097-132.jpg" width="100%" align="left">
        """,
        unsafe_allow_html=True,
    )

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

@st.cache_data(show_spinner=False)
def load_chat_engine():
    return ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY) 

chat_engine = load_chat_engine()

# Aceptar entrada del usuario
if prompt := st.chat_input("Tu consulta"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Limitar la visualización de mensajes al último número determinado para mejorar la velocidad.
last_messages = st.session_state.messages[-10:]

for message in last_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Si el último mensaje no es del asistente, genera una nueva respuesta
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            conversation_string = " ".join([msg['content'] for msg in last_messages])  # Utiliza solo los últimos mensajes para el contexto
            refined_query = query_refiner(conversation_string, prompt)
            
            @st.cache_data(show_spinner=False)
            def get_response(query):
                return get_answer(query)

            response = get_response(refined_query)
            response_content = response["output_text"]
            st.write(response_content)
            message = {"role": "assistant", "content": response_content}
            st.session_state.messages.append(message)
