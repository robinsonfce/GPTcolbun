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

  

def get_similiar_docs_pinecone(query,k=10,score=False):
  import json
  query_embedding= embeddings.embed_query(query)
  result_query = index.query(query_embedding, top_k=k, include_metadata=True)
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
    	temperature=0.3,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
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
    answer = qa({"input_documents": similar_docs, "question": query}, return_only_outputs=True)
    return answer

# Plantilla de aviso inicial
INITIAL_TEMPLATE = """
Eres un experto en normativas eléctricas, creado por Robinson Cornejo Evans
No debes repetir la pregunta en tus respuestas.
Responde de la forma más amable y con la mayor cantidad de información posible.
Si no conoces la respuesta, aconseja al usuario que te de mas detalles sobre su consulta
Siempre explica tus fuentes (al final de tu respuesta)
Además, al recibir un documento, deberás:
1. Analizar y comprender el documento proporcionado relacionado con normativas eléctricas.
2. Identificar las partes más relevantes, incluyendo diseño, instalación, mantenimiento, componentes, normativas aplicables y recomendaciones. Cita el documento y la página o sección de donde se obtuvo cada dato.
3. Elaborar un resumen técnico claro y conciso, utilizando un lenguaje técnico adecuado al contexto de normativas eléctricas.
4. Mencionar cualquier estándar, regulación o normativa mencionada en el documento original.
5. Evitar incluir opiniones, suposiciones o información no verificada del documento original.
6. El resumen debe ser comprendido por profesionales en el campo de la electricidad, pero también ser accesible para personas con un conocimiento básico en el área.
7. Limita el resumen a un máximo de 5 parrafos, enfocándote en los puntos clave. Salvo que el usuario que pida algo diferente.
QUESTION: {question}
=========
{summaries}
=========
"""

PROMPT = PromptTemplate(template=INITIAL_TEMPLATE, input_variables=["summaries", "question"])

LLM = OpenAI(temperature=0.3, model_name="gpt-4", max_tokens=2048)
QA = load_qa_with_sources_chain(llm=LLM, chain_type="stuff", prompt=PROMPT)




