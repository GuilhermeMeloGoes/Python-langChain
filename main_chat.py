import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


load_dotenv()
api_key_google = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.5,
    api_key = api_key_google
)

prompt_sugestao = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um guia de viagens especializado em destinos brasileiros. Apresente-se como Sr. Passeios."),
        ("placeholder", "{historico}"),
        ("human", "{pergunta}")
    ]
)

cadeia = prompt_sugestao | model | StrOutputParser()

memoria = {}
sessao = "aula_langchain_alura" 

def historico_por_sessao(sessao : str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]
    

lista_perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e culturas. Pode sugerir?",
    "Qual a melhor época do ano para visitar esse lugar?"
]

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable = cadeia,
    get_session_history = historico_por_sessao,
    input_messages_key = "pergunta",
    history_messages_key = "historico"
)

for pergunta in lista_perguntas:
    resposta = cadeia_com_memoria.invoke(
        {
            "pergunta": pergunta
        },
        config={
            "configurable": {
                 "session_id": sessao
            }
        }
    )
    print(f"Pergunta usuário: {pergunta}"),
    print(f"Resposta da IA: {resposta} \n")
    