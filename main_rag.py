from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os 

load_dotenv()
api_key_google = os.getenv("GOOGLE_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature=0.5,
    api_key=api_key_google
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=api_key_google
)

arquivos = [
    "documentos/GTB_standard_Nov23.pdf",
     "documentos/GTB_gold_Nov23.pdf",
    "documentos/GTB_platinum_Nov23.pdf"
]

documentos = sum(
    [
        PyPDFLoader(arquivo).load() for arquivo in arquivos
    ], []
)

partes_texto = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
).split_documents(documentos)

dados_recuperados = FAISS.from_documents(
    partes_texto,
    embeddings
).as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o conteúdo fornecido."),
        ("human", "{query}\n\nContexto: \n{contexto}\n\nResposta:")
    ]
)

cadeia = prompt | modelo | StrOutputParser()

def responder(pergunta:str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join(trecho.page_content for trecho in trechos)
    return cadeia.invoke(
        {"query": pergunta,
         "contexto": contexto
        })
    
print(responder("Como devo proceder caso tenha um item comprado roubado e caso eu tenha o cartão gold?"))