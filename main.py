from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

numero_dias = 7
numero_criancas = 2
atividade = "musica"

prompt = f"Crie um roteiro de viagens de {numero_dias} dias, para uma familia com {numero_criancas} crianças, que gosta de {atividade}"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

mensagens = [
    SystemMessage(content="Você é um assistente de viagens especializado em criar roteiro de viagens personalizados."),
    HumanMessage(content=prompt)
]

resposta = llm.invoke(mensagens)

print(resposta.content)