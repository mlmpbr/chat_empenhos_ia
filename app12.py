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

# Configurações
st.set_page_config(
    page_title="Chat com Empenhos",
    page_icon="📊",
    layout="wide",
)
load_dotenv()

# Conexão com o PostgreSQL
def get_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="db_empenhos",
        user="postgres",
        password="sua_senha"  # Preencha aqui
    )

db_uri = os.getenv("DB_URI")
db = SQLDatabase.from_uri(db_uri)

# Inicializar LLM Groq
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# Obter schema da tabela 'despesas'
def get_schema(_):
    return """
    Tabela 'despesas':
    - numNaturezaEmp (VARCHAR): Código da natureza do gasto
    - organograma (VARCHAR): Código do organograma
    - valorEmpenhadoBruto (NUMERIC): Valor total empeado
    - valorLiquidadoBruto (NUMERIC): Valor total liquidado
    - data (DATE): Data do registro
    """

# Executar query e retornar DataFrame
def run_query(query):
    try:
        conn = get_connection()
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Erro na consulta: {str(e)}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

# Gerar SQL a partir da pergunta (sem ```sql)
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
                st.write(f"**Query Gerada:**\n```sql\n{query}\n```")  # Mostrar a query para debug
                
                # Executar a query e obter DataFrame
                df = run_query(query)
                
                if df is not None and not df.empty:
                    # Exibir a tabela
                    st.subheader("Resultados:")
                    st.dataframe(
                        df,
                        column_config={
                            "numNaturezaEmp": "Natureza do Gasto",
                            "data": "Data",
                            "valorEmpenhadoBruto": "Valor Empenhado (R$)",
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