import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# Configurações
st.set_page_config(
    page_title="Chat com Empenhos",
    page_icon="📊",
    layout="wide",
)
load_dotenv()

# Conexão com o PostgreSQL usando SQLAlchemy
def get_connection():
    return create_engine("postgresql://postgres:sua_senha@localhost:5432/db_empenhos")

db_uri = os.getenv("DB_URI")
db = SQLDatabase.from_uri(db_uri)

# Inicializar LLM Groq
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# Obter schema da tabela 'despesas' (ajuste os nomes para minúsculo se necessário)
def get_schema(_):
    return """
    Tabela 'despesas':
    - numnaturezaemp (VARCHAR): Código da natureza do gasto
    - organograma (integer): Código do organograma
    - valorempenhadobruto (NUMERIC): Valor total empeado
    - valorliquidadobruto (NUMERIC): Valor total liquidado
    - data (DATE): Data do registro
    """

# Executar query e retornar DataFrame
def run_query(query):
    try:
        engine = get_connection()
        df = pd.read_sql(query, engine.connect())
        return df
    except Exception as e:
        st.error(f"Erro na consulta: {str(e)}")
        return None

# Gerar SQL a partir da pergunta
template_sql = """Com base no schema da tabela 'despesas' abaixo, escreva uma query SQL que responda à pergunta:

Schema:
{schema}

Pergunta: {question}

SQL Query:
(Retorne APENAS a query SQL, sem ```sql, sem comentários)"""
prompt_sql = ChatPromptTemplate.from_template(template_sql)

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt_sql
    | llm
    | StrOutputParser()
)

# Interface Streamlit
st.title("📊 Chat com Empenhos")
st.write("Pergunte sobre gastos, organogramas ou valores!")

prompt = st.chat_input("Ex: Traga todos o numNaturezaEmp de DEZEMBRO de 2024 por DATA de DEZEMBRO 2024 com os respectivos ValorEmpenhadoBruto - Traga em grafico tipo Tabela")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Processando..."):
            try:
                # Gerar a query
                query = sql_chain.invoke({"question": prompt})
                st.write(f"**Query Gerada:**\n```sql\n{query}\n```")
                
                # Executar a query e obter DataFrame
                df = run_query(query)
                
                if df is not None and not df.empty:
                    # Formatar colunas financeiras para formato brasileiro
                    def format_br(valor):
                        return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    
                    # Aplicar formatação apenas se as colunas existirem
                    if 'valorempenhadobruto' in df.columns:
                        df['valorempenhadobruto'] = df['valorempenhadobruto'].apply(format_br)
                    if 'valorliquidadobruto' in df.columns:
                        df['valorliquidadobruto'] = df['valorliquidadobruto'].apply(format_br)
                    
                    # Exibir a tabela
                    st.subheader("Resultados:")
                    st.dataframe(
                        df,
                        column_config={
                            "numnaturezaemp": "Natureza do Gasto",
                            "data": "Data",
                            "valorempenhadobruto": "Valor Empenhado (R$)",
                            "valorliquidadobruto": "Valor Liquidado (R$)",
                            "identidade": st.column_config.NumberColumn(format="%d"),
                            "idempenho": st.column_config.NumberColumn(format="%d"),
                            "numnaturezaemp": st.column_config.NumberColumn(format="%d"),
                            "organograma": st.column_config.NumberColumn(format="%d")
                        },
                        hide_index=True
                    )
                else:
                    st.warning("Nenhum registro encontrado para os filtros.")
            except Exception as e:
                st.error(f"Erro ao processar a consulta: {str(e)}")