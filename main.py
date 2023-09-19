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

# ... [Todo el código previo que has dado]

# Si el último mensaje no es del asistente, genera una nueva respuesta
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            conversation_string = " ".join([msg['content'] for msg in last_messages])  # Utiliza solo los últimos mensajes para el contexto
            refined_query = query_refiner(conversation_string, prompt)

            @st.cache_data(show_spinner=False)
            def get_response(query):
                return get_answer(query)

            initial_response = get_response(refined_query)
            initial_response_content = initial_response["output_text"]
            
            # Usar la historia, el prompt, y un template para refinar con GPT-3.5-Turbo
            def refiner_answer(context, prompt, initial_answer):
                # Esto es un ejemplo de template, pero puedes modificarlo según tus necesidades.
                template = f"El usuario consultó: \"{prompt}\". En base a los documentos similares, la respuesta inicial es: \"{initial_answer}\". Pero, ¿cuál es la mejor manera de responder a la consulta del usuario teniendo en cuenta el contexto de la conversación?"
                full_query = context + " " + template
                response = chat_engine.get_response(full_query)  # Asumiendo que 'chat_engine' tiene un método 'get_response' que funciona con GPT-3.5-Turbo.
                return response['choices'][0]['text'].strip()  # Puede ser necesario ajustar esta línea según la estructura exacta de la respuesta de GPT-3.5-Turbo.
            
            refined_response = refiner_answer(conversation_string, prompt, initial_response_content)
            
            st.write(refined_response)
            message = {"role": "assistant", "content": refined_response}
            st.session_state.messages.append(message)
