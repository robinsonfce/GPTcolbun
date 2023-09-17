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


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

    __str__ = __repr__


st.markdown("<h2 style='text-align: center;'>Consulta lo que desees sobre seguridad y reglamentos de Colbun</h2>", unsafe_allow_html=True)

st.markdown(
    """
    <img src="https://www.colbun.cl/resourcePackages/colbunweb/assets/dist/images/header/logo.png" width="100" align="middle">
    """,
    unsafe_allow_html=True,
)

st.markdown("<h6 style='text-align: center;'>Creado por: Robinson Cornejo</h6>", unsafe_allow_html=True)


if 'responses' not in st.session_state:
    st.session_state['responses'] = ["En que puedo ayudarte hoy"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []



llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Responda la pregunta con la mayor veracidad posible utilizando el contexto proporcionado,
y si la respuesta no está contenida en el texto a continuación, diga 'Podrias preguntarle al equipo MASSO, seguramente ellos podran orientarte' """)


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)




# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Consulta: ", key="input")
    if query:
        with st.spinner("Escribiendo..."):
            conversation_string = get_conversation_string()
            st.code(conversation_string)
            #user_entities = extract_entities(query)
            #print("#######")
            #print(user_entities)
            refined_query = query_refiner(conversation_string, query)
            #st.subheader("Refined Query:")
            #st.write(refined_query)
            context = get_answer(refined_query)
            print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

       
           

          
