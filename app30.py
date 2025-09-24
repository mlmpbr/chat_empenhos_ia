import os
import streamlit as st
import pandas as pd
import psycopg
from psycopg import Error as PgError
import matplotlib.pyplot as plt
import numpy as np
import re
import traceback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq # Assumindo que esta é a importação correta
from langchain_core.output_parsers import StrOutputParser

# Opcional: Carregar variáveis de ambiente de um arquivo .env
from dotenv import load_dotenv
load_dotenv()

# --- Configurações e Variáveis de Ambiente ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "db_empenhos") # Substitua pelo nome do seu banco de dados
DB_USER = os.getenv("DB_USER", "postgres")     # Substitua pelo seu usuário do banco de dados
DB_URI = os.getenv("DB_URI")
DB_PASSWORD = DB_URI.split(":")[-1].split("@")[0] if DB_URI else os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY or DB_PASSWORD == "your_password":
    st.warning("Por favor, configure suas variáveis de ambiente `GROQ_API_KEY` e `DB_PASSWORD`.")
    st.stop()

# --- Funções de Banco de Dados ---
@st.cache_resource # Cache a conexão para evitar reconexões a cada execução
def get_db_connection():
    """Estabelece e retorna uma conexão com o banco de dados PostgreSQL."""
    try:
        conn = psycopg.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        return conn
    except PgError as e:
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        st.stop() # Interrompe a execução se não conseguir conectar
    except Exception as e:
        st.error(f"Erro inesperado ao conectar ao banco de dados: {e}")
        st.stop()



def run_query(sql_query: str):
    """Executa uma query SQL e retorna os resultados como um DataFrame do Pandas."""
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        df = pd.read_sql_query(sql_query, conn)
        return df
    except PgError as e:
        st.error(f"Erro ao executar a query SQL: {e}")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao executar a query: {e}")
        return None

# --- Funções Auxiliares ---
def extrair_query_sql(resposta_llm: str) -> str:
    """
    Extrai a query SQL de uma string, procurando por blocos de código SQL.
    """
    match = re.search(r"```sql\n(.*?)```", resposta_llm, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"(SELECT .*?;)", resposta_llm, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if 'SELECT' in resposta_llm.upper():
        return resposta_llm.strip()
    return ""

# --- Chain do LangChain ---
@st.cache_resource
def get_sql_chain():
    template_sql = """Sua tarefa é gerar uma única query SQL para PostgreSQL.

Schema da tabela 'despesas':
    - numnaturezaemp (VARCHAR)
    - organograma (VARCHAR)
    - descnaturezaemp (VARCHAR)
    - valorempenhadobruto (NUMERIC)
    - valorliquidadobruto (NUMERIC)
    - data (DATE)
    - identidade (INTEGER)
    - idempenho (INTEGER)

REGRAS RÍGIDAS:
1.  Sua resposta DEVE conter APENAS o código SQL, terminando com ';'.
2.  A coluna de valor agregado (soma, média) DEVE se chamar 'valor'. Ex: SUM(valorempenhadobruto) AS valor.
3.  A coluna de agrupamento (categoria) deve ter um nome simples: 'mes', 'ano', 'descnaturezaemp', etc.
4.  Use sempre nomes de colunas em minúsculas, conforme o schema.

Pergunta do usuário: {question}

SQL Query:
"""
    prompt_sql_template = ChatPromptTemplate.from_template(template_sql)

    sql_chain = (
        sql_chain = get_sql_chain()
    return sql_chain
    