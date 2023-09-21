from langchain.chat_models import ChatOpenAI
import openai
from streamlit_chat import message
from utils import *
from dotenv import load_dotenv
import os
import streamlit as st


donation_link = "https://www.paypal.com/donate/?hosted_button_id=BA335LJJPQM72"


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
    st.markdown("<h6 style='text-align: left;'>Diseñado por <strong>Robinson Cornejo</strong>, Ingeniero Electricista.</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: left;'>Apasionado por la innovación tecnológica y el poder de los datos.</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: left;'><strong>¿Te gusto este proyecto?</strong> ¡Déjame tus comentarios!</h6>", unsafe_allow_html=True)
    st.markdown(
        """
        <h6 style='text-align: left;'>
            <a href="https://www.linkedin.com/in/robinsonfce/" target="_blank">LinkedIn</a> | 
            <a href="mailto:robinson.fce@gmail.com">Correo: robinson.fce@gmail.com</a>
        </h6>
        """, 
        unsafe_allow_html=True
    )
    st.markdown(
    f'<a href="{donation_link}" target="_blank"><img src="https://ginesrom.es/wp-content/uploads/2021/03/Invitame-a-un-cafe-paypal-ginesromero.png" title="Este sitio se mantiene con tu aporte" alt="Donate with PayPal button" width="50%" /></a>',
    unsafe_allow_html=True,
)


# Inicialización de mensajes
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Bienvenido. Soy un asistente digital diseñado para guiarlo a través de pliegos RIC. ¿Qué información o ayuda necesitas?"}
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
            conversation_string = " ".join([msg['content'] for msg in last_messages])  
            refined_query = query_refiner(conversation_string, prompt)

            def get_response(query):
                return get_answer(query)
            
            initial_response = get_response(refined_query)
            initial_response_content = initial_response["output_text"]
            
            # Usar la historia, el prompt, y un template para refinar con GPT-3.5-Turbo
            def refiner_answer(context, prompt, initial_answer):
                template = f"A la consulta: \"{prompt}\"; y en base a los documentos referenciados con la pregunta inicial: \"{initial_answer}\". Responder la consulta, sin dar respuestas fuera del contexto"
                full_query = context + " " + template
                #return response['choices'][0]['text'].strip()
                system_msg='Eres un Ingeniero Eléctrico experto en normativas y estándares eléctricos.
                Tu tarea es proporcionar respuestas detalladas, precisas y con referencias claras a las fuentes, 
                sin repetir la pregunta en tus respuestas.
                Tus respuestas deben estar enfocadas solo a los pliegos RIC.
                Primero responde precisamente lo que se te pregunta y luego agregar un breve comentario. Acota tu respuesta a 100 palabras, salvo que te pidan mas detalles.'

                # Define the user message
                user_msg = full_query
                # Create a dataset using GPT
                response = openai.ChatCompletion.create(model="gpt-3.5-turbo", temperature=0.25,
                                                        messages=[{"role": "system", "content": system_msg},
                                                                  {"role": "user", "content": user_msg}])  
                refined_answer = response["choices"][0]["message"]["content"].strip()
                # Si la respuesta es que no tiene información suficiente, sugiere que te consulte directamente
                if "no tengo suficiente información" in refined_answer.lower():
                    refined_answer = "Lo siento, no tengo suficiente información sobre eso. Por favor, dame mas detalles."
                
                return refined_answer

        
                
            refined_response = refiner_answer(conversation_string, prompt, initial_response_content)
            
            st.write(refined_response)
            message = {"role": "assistant", "content": refined_response}
            st.session_state.messages.append(message)
