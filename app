import os
import pandas as pd
import sqlite3
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import create_engine
from typing import Any
import re

# Function to process CSV as SQLite DB
def process_csv_as_db(csv_file):
    df = pd.read_csv(csv_file.name)
    db_file = "temp_database.db"
    conn = sqlite3.connect(db_file)
    df.to_sql("uploaded_table", conn, index=False, if_exists='replace')
    conn.close()
    return f"sqlite:///{db_file}"

# Function to connect to DB file
def connect_to_db(db_file):
    return f"sqlite:///{db_file.name}"

# Function to process the uploaded file
def upload_and_process_db(file):
    if file.name.endswith(".csv"):
        return process_csv_as_db(file)
    elif file.name.endswith(".db"):
        return connect_to_db(file)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or SQLite DB file.")

# Set up Google API Key
Google_API_KEY = 'AIzaSyAK1PmUoSvsxRvqMRQhGmGvpivhSz_ebdM'
os.environ["GOOGLE_API_KEY"] = Google_API_KEY

# Set up LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.0,
    max_tokens=None,
    google_api_key=Google_API_KEY
)

# Get schema from DB
def get_schema(db):
    if hasattr(db, 'get_table_info'):
        return db.get_table_info()
    elif isinstance(db, dict):
        return db.get('schema', {})
    else:
        raise AttributeError("The provided db object does not contain schema or table information.")

# Prompt template for SQL query generation
template = """Based on the tables schema {schema} below, write a SQLite query that would answer the user's question:
Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)

# SQL generation and validation chains
sql_chain = (
    prompt | llm.bind(stop=["\nSQLResult:"]) | StrOutputParser()
)

query_check_system = """You are a SQL expert with a strong attention to detail. Double check the SQLite query for common mistakes."""
query_check_prompt = ChatPromptTemplate.from_template("{query}\n" + query_check_system)

query_check_chain = (
    query_check_prompt | llm | StrOutputParser()
)

# Query execution tool
def db_query_tool(query: str, db) -> str:
    query = query.replace("```googlesql", "").replace("```", "").strip()
    clean_query_match = re.search(r"(SELECT.*?)(?:;|$)", query, re.DOTALL | re.IGNORECASE)

    if not clean_query_match:
        return "Error: No valid query found starting with 'SELECT'."

    clean_query = clean_query_match.group(1).strip()

    # Execute the clean query
    result = db.run_no_throw(clean_query)

    if not result:
        return "Error: Query failed. Please rewrite your query and try again."

    return result

# Error handling
error_handling_template = """There was an error executing the SQL query: {query}\nError: {error_message}\nPlease rewrite the query to fix the error."""
error_handling_prompt = ChatPromptTemplate.from_template(error_handling_template)

error_handling_chain = (
    error_handling_prompt | llm | StrOutputParser()
)

# Full query chain
def full_query_chain(question: str, db):
    generated_sql = sql_chain.invoke({"question": question, "schema": get_schema(db)})
    validated_sql = query_check_chain.invoke({"query": generated_sql})
    result = db_query_tool(validated_sql, db)

    if "Error" in result:
        fixed_query = error_handling_chain.invoke({"query": validated_sql, "error_message": result})
        result = db_query_tool(fixed_query, db)
        return f"{fixed_query}\n{result}"
    else:
        return f"{validated_sql}\n{result}"

# Gradio app functions
def process_db_upload(file):
    return upload_and_process_db(file)

def query_chain(user_question: str, db_uri: str):
    db = SQLDatabase.from_uri(db_uri)
    return full_query_chain(user_question, db)

# Gradio interface
with gr.Blocks() as interface:
    db_file = gr.File(label="Upload Database File (CSV or .db)")
    db_uri_output = gr.Textbox(label="Database URI", interactive=False)
    
    db_file.change(fn=process_db_upload, inputs=db_file, outputs=db_uri_output)

    user_question_input = gr.Textbox(label="User Question")
    query_output = gr.Textbox(label="Query Result")

    query_button = gr.Button("Generate Query")
    query_button.click(fn=query_chain, inputs=[user_question_input, db_uri_output], outputs=query_output)

# Launch the interface
interface.launch()
