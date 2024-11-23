import os
import uuid
import tempfile
from flask import Flask, render_template, jsonify, request, session
from flask_session import Session
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from src.helper import ConfigManager, serialize_messages, deserialize_messages
from src.model import hf_download_model, hf_download_tokenizer, load_local_model, hf_load_embeddings
from src.prompt import patience_question_prompt

_ = load_dotenv()

config_manager = ConfigManager("config.json")
app_key = os.environ["FLASK_APP_KEY"]

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

app = Flask(__name__)
app.secret_key = app_key

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_FILE_DIR'] = tempfile.gettempdir() 

Session(app)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg

    if 'chat_history' not in session:
        session['chat_history'] = []

    chat_history = deserialize_messages(session['chat_history'])
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="query", output_key="result")
    memory.chat_memory.messages = chat_history
    
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=False,
        chain_type_kwargs=chain_type_kwargs
    )
    
    result = qa.invoke({"query": input})
    answer = result.get("result").split("__\nAnswer:")[-1]

    chat_history = memory.chat_memory.messages
    session['chat_history'] = serialize_messages(chat_history)

    return answer

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )