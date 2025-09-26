import os
import re
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq  # ✅ Nova API

# --- Configurações Iniciais ---
st.set_page_config(
    page_title="Chat com Empenhos",
    page_icon="📊",
    layout="wide",
)
load_dotenv()

# --- Verificação de Variáveis de Ambiente ---
DB_URI = os.getenv("DB_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not DB_URI:
    st.error("ERRO CRÍTICO: A variável de ambiente DB_URI não foi configurada.")
    st.stop()
if not GROQ_API_KEY:
    st.error("ERRO CRÍTICO: A variável de ambiente GROQ_API_KEY não foi configurada.")
    st.stop()

# --- Conexão com o Banco de Dados ---
def get_connection():
    return create_engine(DB_URI)

# --- Schema e Prompt para o LLM ---
def get_schema(_):
    return """
    Tabela 'despesas':
    - numnaturezaemp (VARCHAR)
    - organograma (VARCHAR)
    - descnaturezaemp (VARCHAR)
    - valorempenhadobruto (NUMERIC)
    - valorliquidadobruto (NUMERIC)
    - data (DATE)
    - identidade (INTEGER)
    - idempenho (INTEGER)
    """

template_sql = """Sua tarefa é gerar uma única query SQL para PostgreSQL.

Schema da tabela 'despesas':
{schema}

REGRAS RÍGIDAS:
1.  Sua resposta DEVE conter APENAS o código SQL, terminando com ';'.
2.  A coluna de valor agregado (soma, média) DEVE se chamar 'valor'.
3.  A coluna de agrupamento deve ter um nome simples: 'mes', 'ano', 'descnaturezaemp', etc.
4.  Use sempre nomes de colunas em minúsculas.

Pergunta do usuário: {question}

SQL Query:
"""
prompt_sql_template = ChatPromptTemplate.from_template(template_sql)

# --- Modelo Groq (substitui Gemini) ---
model_a_ser_usado = "llama-3.1-8b-instant"
print(">>> MODELO USADO:", model_a_ser_usado)
print("=====================================================")
print(f"==> DEBUG: PREPARANDO PARA USAR O MODELO: '{model_a_ser_usado}'")
print("=====================================================")

chat_model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt_sql_template
    | chat_model
    | StrOutputParser()
)

# --- Funções Auxiliares (sem alterações) ---
def extrair_query_sql(texto_llm: str) -> str:
    match = re.search(r"SELECT.*?;", texto_llm, re.IGNORECASE | re.DOTALL)
    return match.group(0) if match else None

def run_query(query: str):
    try:
        engine = get_connection()
        with engine.connect() as conn:
            return pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Erro ao executar a consulta SQL: {str(e)}")
        return None

def format_brl(value):
    try:
        if isinstance(value, (int, float)):
            return f"R$ {value:,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")
        return value
    except (ValueError, TypeError):
        return value

def plotar_grafico_barras(df: pd.DataFrame, coluna_categoria: str, coluna_valor: str):
    st.subheader("Resultado em Gráfico de Barras")
    plt.style.use('seaborn-v0_8-whitegrid')
    df = df.sort_values(by=coluna_categoria).reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(df[coluna_categoria]):
        df[coluna_categoria] = df[coluna_categoria].astype(int).astype(str)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df[coluna_categoria], df[coluna_valor], color='#3399FF', label='Valor Total', width=0.6)
    if len(df.index) > 1:
        x = np.arange(len(df[coluna_categoria])); y = df[coluna_valor].astype(float); z = np.polyfit(x, y, 1); p = np.poly1d(z)
        ax.plot(x, p(x), "r--", label="Linha de Tendência")
    ax.set_title(f'Total por {coluna_categoria.capitalize()}', fontsize=16)
    ax.set_xlabel(coluna_categoria.capitalize(), fontsize=12)
    ax.set_ylabel('Valor (R$)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"R$ {int(val):,} ".replace(",", ".")))
    ax.legend()
    limite_superior = df[coluna_valor].max() * 1.15
    ax.set_ylim(0, limite_superior)
    plt.xticks(rotation=45, ha='right'); plt.tight_layout(); st.pyplot(fig, use_container_width=True)

def plotar_grafico_pizza(df: pd.DataFrame, coluna_labels: str, coluna_valores: str):
    st.subheader("Resultado em Gráfico de Pizza")
    df = df.sort_values(by=coluna_labels).reset_index(drop=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    if pd.api.types.is_numeric_dtype(df[coluna_labels]):
        df[coluna_labels] = df[coluna_labels].astype(int).astype(str)
    explode = [0] * len(df)
    maior_fatia_idx = df[coluna_valores].idxmax()
    if maior_fatia_idx in df.index:
        explode[df.index.get_loc(maior_fatia_idx)] = 0.1
    fig, ax = plt.subplots(figsize=(8, 4))
    wedges, _, autotexts = ax.pie(df[coluna_valores], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.axis('equal'); ax.set_title(f'Distribuição por {coluna_labels.capitalize()}', fontsize=16)
    ax.legend(wedges, df[coluna_labels], title=coluna_labels.capitalize(), loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize='small')
    plt.setp(autotexts, size=8, weight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# --- Interface Streamlit (sem alterações) ---
st.title("📊 Chat com Empenhos")
st.caption("Faça uma pergunta sobre os dados de empenhos. Peça totais, médias ou gráficos.")

if prompt := st.chat_input("Ex: traga o total empenhado por mês em 2024"):
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analisando sua pergunta e gerando a consulta..."):
            try:
                resposta_llm = sql_chain.invoke({"question": prompt})
                sql_query = extrair_query_sql(resposta_llm)

                if sql_query:
                    df_resultado = run_query(sql_query)

                    if df_resultado is not None and not df_resultado.empty:
                        is_chart_request = any(word in prompt.lower() for word in ['gráfico', 'grafico', 'pizza', 'barras', 'chart', 'comparação'])

                        if is_chart_request and df_resultado.shape[1] == 2 and 'valor' in df_resultado.columns:
                            coluna_categoria = [col for col in df_resultado.columns if col != 'valor'][0]
                            if 'pizza' in prompt.lower():
                                plotar_grafico_pizza(df_resultado, coluna_labels=coluna_categoria, coluna_valores='valor')
                            else:
                                plotar_grafico_barras(df_resultado, coluna_categoria=coluna_categoria, coluna_valor='valor')

                            st.write(f"**Dados da Consulta ({coluna_categoria.capitalize()}):**")
                            df_formatado_tabela = df_resultado.copy()
                            df_formatado_tabela['valor'] = df_formatado_tabela['valor'].apply(format_brl)
                            st.dataframe(df_formatado_tabela, hide_index=True, use_container_width=True)
                        else:
                            st.subheader("Resultado da Consulta")
                            df_formatado = df_resultado.copy()
                            for col in df_formatado.select_dtypes(include=np.number).columns:
                                if 'valor' in col or 'empenho' in col or 'liquido' in col:
                                    df_formatado[col] = df_formatado[col].apply(format_brl)
                            st.dataframe(df_formatado, hide_index=True, use_container_width=True)

                    elif df_resultado is not None:
                        st.warning("A consulta foi executada, mas não retornou resultados.")

            except Exception as e:
                st.error(f"Ocorreu um erro inesperado durante o processamento: {str(e)}")