import pinecone
import openai
import streamlit as st

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate


import os
from dotenv import load_dotenv

# Define las variables de entorno y otros secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
#--------------------------------------------------------

model_name = "text-embedding-ada-002"

pinecone.init(      
	api_key=PINECONE_API_KEY,      
	environment='us-west4-gcp'      
)      
index = pinecone.Index('ric')
index_pinecone_index          = pinecone.Index(index_name='ric')
embeddings = OpenAIEmbeddings()
#--------------------------------------------------------



class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

    __str__ = __repr__

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
  result_query = index_pinecone_index.query(query_embedding, top_k=k, include_metadata=True)
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

##llm = OpenAI(model_name=model_name)
##chain = load_qa_chain(llm, chain_type="stuff")


initial_template = """
You are an expert in occupational safety and protocols and you represent the company Colbun S.A.
You should not repeat the question in your answers.
Respond in the kindest way and with as much information as possible.
If you do not know the answer, advise the user to consult their boss or a MASSO advisor.
always explain your sources
QUESTION: {question}
=========
{summaries}
=========

"""


PROMPT = PromptTemplate(template=initial_template, input_variables=["summaries", "question"])


llm = OpenAI(temperature=0.3, model_name="gpt-3.5-turbo-0301", max_tokens=512)
qa = load_qa_with_sources_chain(llm=llm, chain_type="stuff", prompt=PROMPT)

"""
def get_answer(query):
  similar_docs = get_similiar_docs_pinecone(query)
  print(similar_docs)
  print('------')
  #answer = chain.run(input_documents=similar_docs, question=query)
  answer =  qa({"input_documents": similar_docs, "question": query}, return_only_outputs=True)
  return answer

"""


def get_answer(query):
  similar_docs = get_similiar_docs_pinecone(query)
  print(similar_docs)
  print('------')
  answer =  qa({"input_documents": similar_docs, "question": query}, return_only_outputs=True)
  return answer


def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Dada la consulta del usuario y el historial de la conversación, tu objetivo es formular una pregunta más refinada y específica que te permita obtener la información más relevante de la base de conocimientos para responder de la mejor manera posible.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.5,
    max_tokens=512,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

"""
def query_refiner(conversation, query):
    user_entities = extract_entities(query)
    refined_query = query + " " + " ".join(user_entities)
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Dada la consulta del usuario, las entidades identificadas y el historial de la conversación, tu objetivo es formular una pregunta más refinada y específica que te permita obtener la información más relevante de la base de conocimientos para responder de la mejor manera posible.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {refined_query}\n\nRefined Query:",
    temperature=0.5,
    max_tokens=512,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']
"""
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

