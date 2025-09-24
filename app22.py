import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# --- Configurações Iniciais ---
st.set_page_config(
    page_title="Chat com Empenhos",
    page_icon="📊",
    layout="wide",
)
load_dotenv()

# --- Conexão com o Banco de Dados ---
def get_connection():
    return create_engine("postgresql://postgres:sua_senha@localhost:5432/db_empenhos")

# --- Schema e Prompt para o LLM ---
def get_schema(_):
    return """
    Tabela 'despesas':
    - numnaturezaemp (VARCHAR): Código da natureza do gasto
    - organograma (VARCHAR): Código do organograma
    - descnaturezaemp (VARCHAR): Descrição da natureza do gasto
    - valorempenhadobruto (NUMERIC): Valor total empenhado
    - valorliquidadobruto (NUMERIC): Valor total liquidado
    - data (DATE): Data do registro (formato: YYYY-MM-DD)
    - identidade (integer): ID do registro
    - idempenho (integer): ID do empenho
    """

# V.4 - PROMPT AINDA MAIS COMPLETO E EXPLÍCITO
template_sql = """Com base no schema da tabela 'despesas' abaixo, escreva uma query SQL que responda à pergunta.

Schema:
{schema}

REGRAS OBRIGATÓRIAS:
1.  Use sempre o nome exato das colunas do schema (ex: descnaturezaemp).
2.  Para qualquer pedido de agregação (soma, média), a coluna com o valor calculado DEVE OBRIGATORIAMENTE se chamar 'valor'. Ex: SUM(valorempenhadobruto) AS valor.
3.  A coluna de agrupamento (categoria) deve ter um nome simples e direto:
    - Agrupado por mês -> a coluna DEVE se chamar 'mes'. Ex: EXTRACT(MONTH FROM data) AS mes.
    - Agrupado por ano -> a coluna DEVE se chamar 'ano'. Ex: EXTRACT(YEAR FROM data) AS ano.
    - Agrupado por categoria -> a coluna DEVE se chamar 'descnaturezaemp'.
    - Agrupado por organograma -> a coluna DEVE se chamar 'organograma'.
4.  A query deve retornar APENAS a query SQL, sem ```sql, sem comentários e terminando com ponto e vírgula ';'.

Pergunta: {question}

SQL Query:
"""
prompt_sql_template = ChatPromptTemplate.from_template(template_sql)

# --- Chain do LangChain ---
sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt_sql_template
    | ChatGroq(
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192"
    )
    | StrOutputParser()
)

# --- Funções Auxiliares ---
def run_query(query: str):
    try:
        engine = get_connection()
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df
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

# --- Funções de Plotagem Genéricas ---

# V.6 - FUNÇÃO DE GRÁFICO DE BARRAS GENÉRICA
def plotar_grafico_barras(df: pd.DataFrame, coluna_categoria: str, coluna_valor: str):
    st.subheader("Resultado em Gráfico de Barras")
    
    # V.7 - CORREÇÃO DE FORMATAÇÃO (ex: 2022.0 -> 2022, 1.0 -> 1)
    # Garante que os rótulos do eixo X sejam strings limpas
    if pd.api.types.is_numeric_dtype(df[coluna_categoria]):
        df[coluna_categoria] = df[coluna_categoria].astype(int).astype(str)
        
    df = df.sort_values(by=coluna_categoria).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df[coluna_categoria], df[coluna_valor], color='#66b3ff', label='Valor Total')

    # Linha de tendência
    if len(df.index) > 1:
        x = df.index
        y = df[coluna_valor]
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(df[coluna_categoria], p(x), "r--", label="Linha de Tendência")

    ax.set_title(f'Total por {coluna_categoria.capitalize()}', fontsize=16)
    ax.set_xlabel(coluna_categoria.capitalize(), fontsize=12)
    ax.set_ylabel('Valor (R$)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"R$ {int(val):,} ".replace(",", ".")))
    ax.legend()
    plt.xticks(rotation=45) # Rotação para melhor visualização de textos longos
    plt.tight_layout()
    st.pyplot(fig)

# V.6 - FUNÇÃO DE GRÁFICO DE PIZZA GENÉRICA
def plotar_grafico_pizza(df: pd.DataFrame, coluna_labels: str, coluna_valores: str):
    st.subheader("Resultado em Gráfico de Pizza")
    
    # V.7 - CORREÇÃO DE FORMATAÇÃO
    if pd.api.types.is_numeric_dtype(df[coluna_labels]):
        df[coluna_labels] = df[coluna_labels].astype(int).astype(str)

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, _, autotexts = ax.pie(
        df[coluna_valores],
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.Paired.colors
    )
    ax.axis('equal')
    ax.set_title(f'Distribuição por {coluna_labels.capitalize()}', fontsize=16)
    ax.legend(wedges, df[coluna_labels], title=coluna_labels.capitalize(), loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=8, weight="bold")
    st.pyplot(fig)

# --- Interface Principal do Streamlit ---
st.title("📊 Chat com Empenhos")
st.caption("Faça uma pergunta sobre os dados de empenhos. Peça totais, médias ou gráficos.")

if prompt := st.chat_input("Ex: Compare os totais por ano em um gráfico de pizza"):
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analisando sua pergunta e buscando os dados..."):
            try:
                sql_query = sql_chain.invoke({"question": prompt})
                st.write("🔍 **Query Gerada:**")
                st.code(sql_query, language="sql")

                df_resultado = run_query(sql_query)

                if df_resultado is not None and not df_resultado.empty:
                    
                    # V.5 - LÓGICA DE PLOTAGEM GENÉRICA E FLEXÍVEL
                    is_chart_request = any(word in prompt.lower() for word in ['gráfico', 'grafico', 'pizza', 'barras', 'chart', 'comparação'])
                    
                    # Um gráfico simples geralmente tem 2 colunas: uma categoria e um valor.
                    if is_chart_request and df_resultado.shape[1] == 2 and 'valor' in df_resultado.columns:
                        
                        # Identifica qual coluna é a de categoria (a que não se chama 'valor')
                        coluna_categoria = [col for col in df_resultado.columns if col != 'valor'][0]
                        
                        if 'pizza' in prompt.lower():
                            plotar_grafico_pizza(df_resultado, coluna_labels=coluna_categoria, coluna_valores='valor')
                        else: # Padrão é barras
                            plotar_grafico_barras(df_resultado, coluna_categoria=coluna_categoria, coluna_valor='valor')

                        # Sempre exibe a tabela de dados junto com o gráfico
                        st.write(f"**Dados da Consulta ({coluna_categoria.capitalize()}):**")
                        df_formatado_tabela = df_resultado.copy()
                        df_formatado_tabela['valor'] = df_formatado_tabela['valor'].apply(format_brl)
                        st.dataframe(df_formatado_tabela, hide_index=True, use_container_width=True)

                    else: # Se não for um gráfico ou a query for complexa (mais de 2 colunas)
                        st.subheader("Resultado da Consulta")
                        df_formatado = df_resultado.copy()
                        for col in df_formatado.select_dtypes(include=np.number).columns:
                            if 'valor' in col or 'empenho' in col or 'liquido' in col:
                                df_formatado[col] = df_formatado[col].apply(format_brl)
                        st.dataframe(df_formatado, hide_index=True, use_container_width=True)
                
                elif df_resultado is not None:
                     st.warning("A consulta não retornou resultados. Tente refazer sua pergunta.")

            except Exception as e:
                st.error(f"Ocorreu um erro inesperado durante o processamento: {str(e)}")