from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import asyncio
import os

load_dotenv()
api_key_google = os.getenv("GOOGLE_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.5,
    api_key = api_key_google
)


prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sra. Praia. Você é uma especialista em viagens com destinos para a praia."),
        ("human", "{perguntas}")
    ]
)

prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sr. Montanha. Você é uma especialista em viagens com destinos para a montanha e atividades radicais."),
        ("human", "{perguntas}")
    ]
)


cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()

class Rota(BaseModel):
    destino: Literal["praia", "montanha"] = Field(
        description="Classifique o destino 'praia' ou 'montanha'."
    )
    
parser = StrOutputParser(pydantic_object=Rota)
    
prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda com 'praia' ou 'montanha'"),
        ("human", "{perguntas}")
    ]
)

roteador = prompt_roteador | modelo.with_structured_output(Rota)

class Estado(TypedDict):
    pergunta: str
    destino: Rota
    resposta: str


async def no_roteador(estado: Estado, config=RunnableConfig):
    return {"destino": await roteador.ainvoke({"perguntas": estado["pergunta"]}, config)}

async def no_praia(estado: Estado, config=RunnableConfig):
    return {"resposta": await cadeia_praia.ainvoke({"perguntas": estado["pergunta"]}, config)}

async def no_montanha(estado: Estado, config=RunnableConfig):
    return {"resposta": await cadeia_montanha.ainvoke({"perguntas": estado["pergunta"]}, config)}

def escolher_no(estado:Estado)->Literal["praia", "montanha"]:
    return "praia" if estado["destino"].destino == "praia" else "montanha"

grafo = StateGraph(Estado) 

grafo.add_node("roteador", no_roteador)
grafo.add_node("praia", no_praia)
grafo.add_node("montanha", no_montanha)

grafo.add_edge(START, "roteador")
grafo.add_conditional_edges("roteador", escolher_no)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()

async def main():
    resposta = await app.ainvoke(
        {
            "pergunta": "Quero visitar um lugar no Brasil famoso por praias e cultura"
        }
    )
    print(resposta["resposta"])
    
asyncio.run(main())