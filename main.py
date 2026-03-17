from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.globals import set_debug
import os


set_debug(True)

load_dotenv()
api_key_google = os.getenv("GOOGLE_API_KEY")

class Destino(BaseModel):
    cidade:str = Field("A cidade recomendada para visitar")
    motivo: str = Field("O motivo pelo qual é interessante visitar essa cidade")

class Restaurantes(BaseModel):
    cidade:str = Field("A cidade recomendada para visitar")
    restaurantes:str = Field("Restaurantes recomendados na cidade")

parser_destino = JsonOutputParser(pydantic_object=Destino)
parser_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

prompt_cidade = PromptTemplate(
    template = """
        Sugira uma cidade dado o meu interesse por {interesse}.
        {formato_de_saida}
    """,
    input_variables = ["interesse"],
    partial_variables = {"formato_de_saida": parser_destino.get_format_instructions()}
)

prompt_restaurantes = PromptTemplate(
    template = """
        Sugira restaurantes populares entre locais em {cidade}.
        {formato_de_saida}
    """,
    partial_variables = {"formato_de_saida": parser_restaurantes.get_format_instructions()}
)

prompt_cultural = PromptTemplate(
    template="""
        Sugira atividades e locais culturais em {cidade}
    """
)

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.5,
    api_key = api_key_google
)

cadeia_cidade = prompt_cidade | llm | parser_destino
cadeia_restaurantes = prompt_restaurantes | llm | parser_restaurantes
cadeia_cultural = prompt_cultural | llm | StrOutputParser()

cadeia = (cadeia_cidade | cadeia_restaurantes | cadeia_cultural)

resposta = cadeia.invoke(
    {
        "interesse": "praias"
    }
)

print(resposta)