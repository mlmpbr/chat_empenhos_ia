# 📊 Chat com Empenhos - Análise de Dados Financeiros com IA

Este projeto é uma aplicação web construída com Streamlit que permite aos usuários analisarem dados de empenhos financeiros de forma interativa. Utilizando um modelo de linguagem avançado (Google Gemini), a aplicação traduz perguntas em linguagem natural para consultas SQL, executando-as em um banco de dados PostgreSQL e exibindo os resultados em tabelas e gráficos.

## ✨ Funcionalidades

- **Chat Interativo:** Faça perguntas em português sobre os dados (ex: "qual o total empenhado por mês?").
- **Geração Automática de SQL:** A IA converte a pergunta do usuário em uma consulta SQL válida.
- **Visualização de Dados:** Exibe os resultados em tabelas formatadas e gera gráficos de barras ou pizza automaticamente.
- **Pipeline de Dados Seguro:** Utiliza um arquivo `.env` para gerenciar chaves de API e credenciais de banco de dados de forma segura.

## 🏛️ Arquitetura

A solução segue um pipeline de dados moderno:

1.  **Frontend (Streamlit):** Captura a pergunta do usuário.
2.  **IA (LangChain + Google Gemini):** Recebe a pergunta e o schema do banco, gerando uma consulta SQL.
3.  **Backend (Python + SQLAlchemy):** Executa a consulta SQL no banco de dados.
4.  **Banco de Dados (PostgreSQL):** Armazena todos os dados de empenhos.
5.  **Visualização (Streamlit + Matplotlib):** Renderiza os resultados em tabelas e gráficos.

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python
- **Framework Web:** Streamlit
- **IA e Orquestração:** LangChain, Google Gemini API
- **Banco de Dados:** PostgreSQL
- **Manipulação de Dados:** Pandas
- **Gráficos:** Matplotlib
- **Conexão com DB:** SQLAlchemy, Psycopg2

## 🚀 Como Executar o Projeto Localmente

Siga os passos abaixo para configurar e executar o projeto.

1.  **Clone o Repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/nome-do-seu-repo.git](https://github.com/seu-usuario/nome-do-seu-repo.git)
    cd nome-do-seu-repo
    ```

2.  **Crie e Ative o Ambiente Conda:**
    ```bash
    conda create --name chat_empenhos python=3.9
    conda activate chat_empenhos
    ```

3.  **Instale as Dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure o Banco de Dados:**
    - Certifique-se de ter uma instância do PostgreSQL ativa.
    - Crie um banco de dados (ex: `db_empenhos`).
    - Crie a tabela `despesas` e popule com seus dados.

5.  **Configure as Variáveis de Ambiente:**
    - Crie um arquivo chamado `.env` na raiz do projeto.
    - Copie o conteúdo do arquivo `.env.example` e preencha com suas credenciais:
    ```
    DB_URI="postgresql+psycopg2://USUARIO:SENHA@HOST:PORTA/NOME_DO_BANCO"
    GOOGLE_API_KEY="sua_chave_de_api_do_google"
    ```

6.  **Execute a Aplicação:**
    ```bash
    streamlit run app.py
    ```

## ✍️ Autor

- **[Mario Mello](https://www.linkedin.com/in/mariomello8/)**