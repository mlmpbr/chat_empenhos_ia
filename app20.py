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
import matplotlib.pyplot as plt

# Configurações
st.set_page_config(
    page_title="Chat com Empenhos",
    page_icon="📊",
    layout="wide",
)
load_dotenv()

# Função para executar query e retornar DataFrame - MOVIDA PARA CIMA
def run_query(query):
    try:
        engine = get_connection()
        df = pd.read_sql(query, engine.connect())
        return df
    except Exception as e:
        st.error(f"Erro na consulta: {str(e)}")
        return None

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

# Schema atualizado com data como DATE
def get_schema(_):
    return """
    Tabela 'despesas':
    - numnaturezaemp (VARCHAR): Código da natureza do gasto
    - organograma (integer): Código do organograma
    - valorempenhadobruto (NUMERIC): Valor total empeado
    - valorliquidadobruto (NUMERIC): Valor total liquidado
    - data (DATE): Data do registro (formato: YYYY-MM-DD)
    - identidade (integer): ID do registro
    - idempenho (integer): ID do empehno
    """

# Template SQL corrigido para "todos os dados"
template_sql = """Com base no schema da tabela 'despesas' abaixo, escreva uma query SQL que responda à pergunta:

Schema:
{schema}

IMPORTANTE:
- Se a pergunta contiver "total", "soma", "valor total", etc., use funções de agregação (SUM, COUNT)
- Para valores grandes, sempre retorne NUMERIC para evitar overflow
- Para perguntas sobre meses/anos específicos, use EXTRACT(MONTH/YEAR FROM data)
- Para formatar valores no padrão brasileiro, use a função format_br nos resultados

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

# Função para formatar valores em Real Brasileiro (Padrão: 1.234.567,89)
def format_br(valor):
    try:
        if isinstance(valor, (int, float)):
            # Converte para string com 2 casas decimais
            formatted = f"{valor:,.2f}"
            # Substitui para formato brasileiro
            return formatted.replace(",", "X").replace(".", ",").replace("X", ".")
        return str(valor)
    except:
        return str(valor)

# Interface Streamlit
st.title("📊 Chat com Empenhos")
st.write("Pergunte sobre gastos, organogramas ou valores!")

prompt = st.chat_input("Ex: Qual o total de empehno de JANEIRO de 2024?")

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
                    # Criar cópia para formatação
                    display_df = df.copy()
                    
                    # Identificar colunas numéricas para formatação
                    numeric_cols = display_df.select_dtypes(include=['number']).columns.tolist()
                    
                    # Aplicar formatação em todas as colunas numéricas
                    for col in numeric_cols:
                        if col in ['valorempenhadobruto', 'valorliquidadobruto']:
                            display_df[col] = display_df[col].apply(format_br)
                        else:
                            # Para colunas de ID ou códigos, manter como estão
                            pass
                    
                    # Exibir a tabela
                    st.subheader("Resultados:")
                    st.dataframe(
                        display_df,
                        column_config={
                            col: st.column_config.NumberColumn(format="%d") 
                            for col in numeric_cols 
                            if col not in ['valorempenhadobruto', 'valorliquidadobruto', 'numnaturezaemp']
                        },
                        hide_index=True
                    )
                    
                    # Verificar se o usuário solicitou gráfico
                    if any(word in prompt.lower() for word in ['gráfico', 'grafico', 'chart', 'plot']):
                        # Identificar colunas para o gráfico
                        if len(df.columns) >= 2:
                            category_col = df.columns[0]
                            value_col = None
                            
                            # Encontrar a primeira coluna numérica após a categórica
                            for col in df.columns[1:]:
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    value_col = col
                                    break
                            
                            if value_col:
                                # Verificar se é numérico
                                if pd.api.types.is_numeric_dtype(df[value_col]):
                                    # Criar figura com tamanho maior para melhor visualização
                                    plt.figure(figsize=(12, 10))
                                    
                                    # Criar gráfico de pizza com ajustes para melhor legibilidade
                                    if any(word in prompt.lower() for word in ['pizza', 'pie']):
                                        st.subheader("Gráfico de Pizza")
                                        
                                        # Criar figura
                                        fig, ax = plt.subplots(figsize=(12, 10))
                                        
                                        # Calcular percentuais
                                        total = df[value_col].sum()
                                        percentages = df[value_col] / total * 100
                                        
                                        # Plotar o gráfico de pizza
                                        wedges, labels, autotexts = ax.pie(
                                            df[value_col], 
                                            labels=None,  # Não colocar labels nas fatias
                                            autopct='%1.1f%%',
                                            wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
                                        )
                                        
                                        # Criar legenda com labels e percentuais
                                        legend_labels = [
                                            f"{category}: {percent:.1f}%" 
                                            for category, percent in zip(df[category_col], percentages)
                                        ]
                                        
                                        # Adicionar legenda
                                        ax.legend(
                                            wedges, 
                                            legend_labels,
                                            title=f"{category_col} - {value_col}",
                                            loc="center left",
                                            bbox_to_anchor=(1, 0, 0.5, 1),
                                            fontsize=10
                                        )
                                        
                                        # Ajustar layout
                                        plt.tight_layout()
                                        
                                        # Exibir no Streamlit
                                        st.pyplot(fig)
                                    
                                    # Criar gráfico de barras
                                    elif any(word in prompt.lower() for word in ['barras', 'bar']):
                                        st.subheader("Gráfico de Barras")
                                        
                                        # Criar figura
                                        fig, ax = plt.subplots(figsize=(12, 8))
                                        
                                        # Plotar barras
                                        df.set_index(category_col)[value_col].plot(kind='bar', ax=ax)
                                        
                                        # Adicionar rótulos e título
                                        ax.set_ylabel(value_col)
                                        ax.set_xlabel(category_col)
                                        ax.set_title(f'Gráfico de Barras: {category_col} vs {value_col}')
                                        
                                        # Rotacionar labels para melhor visualização
                                        plt.xticks(rotation=45, ha='right')
                                        
                                        # Ajustar layout
                                        plt.tight_layout()
                                        
                                        # Exibir no Streamlit
                                        st.pyplot(fig)
                            else:
                                st.warning("Não foi possível identificar uma coluna numérica para o gráfico")
                    
                else:
                    st.warning("Nenhum registro encontrado para os filtros.")
            except Exception as e:
                st.error(f"Erro ao processar a consulta: {str(e)}")