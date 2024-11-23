import os
from flask import Flask, render_template, jsonify, request
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.helper import ConfigManager
from src.model import hf_download_model, hf_download_tokenizer, load_local_model, hf_load_embeddings
from src.prompt import patience_question_prompt

_ = load_dotenv()

config_manager = ConfigManager("config.json")

model = hf_download_tokenizer(config_manager)
tokenizer = hf_download_tokenizer(config_manager)
llm = load_local_model(config_manager, tokenizer)
embeddings  =   hf_load_embeddings(config_manager)

prompt_template = PromptTemplate(
    template=patience_question_prompt, 
    input_variables=["context", "question"]
)
chain_type_kwargs={"prompt": prompt_template}

pc = Pinecone(api_key=os.getenv("PINECONE_TOKEN"))
index = pc.Index(config_manager.get_config("pinecone_index_name"))
vector_db = PineconeVectorStore(index=index, embedding=embeddings)

qa=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    result = qa({"query": input})
    answer = result.get("result").split("__\nAnswer:")[-1]
    return answer

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )