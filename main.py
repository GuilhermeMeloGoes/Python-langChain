from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key_google = os.getenv("GOOGLE_API_KEY")

numero_dias = 7
numero_criancas = 2
atividade = "praia"

modelo_prompt = PromptTemplate(
    template = '''
        Crie um roteiro de viagem de {dias} dias,
        para uma família com {numero_criancas} crianças,
        que gostam de {atividade}.
    '''
)

prompt = modelo_prompt.format(
    dias = numero_dias,
    numero_criancas = numero_criancas,
    atividade = atividade  
)

print("Prompt: \n", prompt)

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.5,
    api_key = api_key_google
)

mensagens = [
    SystemMessage(content="Você é um assistente de viagens especializado em criar roteiro de viagens personalizados."),
    HumanMessage(content=prompt)
]

resposta = llm.invoke(mensagens)

print(resposta.content)