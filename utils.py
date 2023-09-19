# Importaciones de servicios y bibliotecas
import json
import os
import openai
import pinecone
import streamlit as st
from dotenv import load_dotenv
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# Configuración y constantes globales
load_dotenv()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

MODEL_NAME = "text-embedding-ada-002"
pinecone.init(api_key=PINECONE_API_KEY, environment='us-west4-gcp')
INDEX_PINECONE = pinecone.Index(index_name='codegpt')
EMBEDDINGS = OpenAIEmbeddings()

# Clase para representar un documento
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

    __str__ = __repr__

# Función para transformar un diccionario en un Document
def transform_dict_to_document(dict_list):
    document_list = []
    
    for dict_obj in dict_list:
        # Extraer el contenido de la página y los metadatos del diccionario
        page_content = dict_obj['metadata']['text']
        page_content = page_content.replace('\n', '')  # Elimina los saltos de línea

        metadata = {'page': dict_obj['metadata']['page'], 'source': dict_obj['metadata']['source']}

        # Crear un Document con el contenido de la página y los metadatos
        doc = Document(page_content=page_content, metadata=metadata)

        # Añadir el Document a la lista
        document_list.append(doc)

    # Devolver la lista de Documents
    return document_list

  

def get_similiar_docs_pinecone(query,k=3,score=False):
  import json
  query_embedding= EMBEDDINGS.embed_query(query)
  result_query = INDEX_PINECONE.query(query_embedding, top_k=k, include_metadata=True)
  result_query_json=json.dumps(result_query.to_dict())

  def json_to_list(json_string):
    # Convertir la cadena de caracteres a diccionario
    json_string = json_string.replace("'", '"')  # JSON necesita comillas dobles
    json_dict = json.loads(json_string)
    
    # Extraer 'matches' que es una lista de diccionarios
    matches_list = json_dict['matches']

    return matches_list

  similar_docs=transform_dict_to_document(json_to_list(result_query_json))

  return similar_docs



# Función para refinar una consulta dada una conversación anterior
def query_refiner(conversation, query):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Dada la consulta del usuario y el historial de la conversación, tu objetivo es formular una pregunta más refinada y específica centrada en el área de normativas eléctricas. Esta pregunta refinada debe ayudarte a obtener la información más relevante de la base de conocimientos para responder de la mejor manera posible. La consulta refinada debe estar en forma de pregunta y no exceder de 2 oraciones.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    	temperature=0.2,
        max_tokens=512,
        top_p=0.75,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )
    return response['choices'][0]['text']

# Función para obtener el historial de conversación
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

# Función para obtener una respuesta a una consulta
def get_answer(query):
    similar_docs = get_similiar_docs_pinecone(query)
    print(similar_docs)
    print('------')
    answer = QA({"input_documents": similar_docs, "question": query}, return_only_outputs=True)
    return answer





# Plantilla de aviso inicial

INITIAL_TEMPLATE = """
Eres un especialista en normativas eléctricas actuando en representación de Robinson Cornejo Evans, Instalador Electricista Clase A.

No repitas la pregunta en tus respuestas.
Brinda respuestas amables, detalladas e informativas.
Si desconoces algo, sugiere al usuario que te consulte directamente.
Siempre cita tus fuentes, especificando el documento y la sección o página correspondiente.
Al recibir un documento, actúa de la siguiente manera:

Analiza y entiende el contenido relacionado con normativas eléctricas.
Destaca aspectos cruciales como diseño, instalación, mantenimiento, componentes, normativas y recomendaciones.
Elabora un resumen técnico claro, conciso y con el lenguaje técnico pertinente. Asegúrate de citar la fuente y la ubicación exacta de cada dato.
Indica todos los estándares, regulaciones o normativas mencionadas en el documento original.
Abstente de incluir opiniones, suposiciones o información no verificada.
Tu resumen debe ser comprensible tanto para profesionales eléctricos como para aquellos con conocimientos básicos en el área, siempre poniendo énfasis en los puntos clave.

QUESTION: {question}
=========
{summaries}
=========
"""


PROMPT = PromptTemplate(template=INITIAL_TEMPLATE, input_variables=["summaries", "question"])

LLM = OpenAI(temperature=0.3, model_name="gpt-4", max_tokens=2048)
QA = load_qa_with_sources_chain(llm=LLM, chain_type="stuff", prompt=PROMPT)