import os
import tempfile
from flask import Flask, render_template, request, session
from flask_session import Session
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from pinecone import Pinecone
from src.helper import ConfigManager, serialize_messages, deserialize_messages, remove_all_in_directory
from src.model import hf_download_model, hf_download_tokenizer, load_local_model, hf_load_embeddings
from src.prompt import qa_prompt, output_cleanup_prompt
from pinecone import Pinecone
import logger # To initialize and generate logs


# Initialize configuration manager and app key
config_manager = ConfigManager("config.json")
app_key = os.environ["FLASK_APP_KEY"]

# Remove existing model directories
remove_all_in_directory(config_manager.get_config("model_directory"))

# Download and load the model, tokenizer, and embeddings
model = hf_download_model(config_manager)
tokenizer = hf_download_tokenizer(config_manager)
llm = load_local_model(config_manager, tokenizer)
embeddings = hf_load_embeddings(config_manager)

# Create the question-answering chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Initialize Pinecone and the vector store
pc = Pinecone(api_key=os.getenv("PINECONE_TOKEN"))
index = pc.Index(config_manager.get_config(["pinecone", "index_name"]))
vector_db = PineconeVectorStore(index=index, embedding=embeddings)

# Create the retrieval chain
rag_chain = create_retrieval_chain(
    vector_db.as_retriever(search_kwargs={"k": 3}),
    question_answer_chain
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = app_key

# Configure session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_FILE_DIR'] = tempfile.mkdtemp()

Session(app)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    user_input = request.form["msg"]

    if 'chat_history' not in session:
        session['chat_history'] = []

    chat_history_messages = deserialize_messages(session['chat_history'])

    chat_message_history = ChatMessageHistory()
    chat_message_history.messages = chat_history_messages

    memory = ConversationBufferWindowMemory(chat_memory=chat_message_history, return_messages=True, k=3)

    # Get the result from the RAG chain
    result = rag_chain.invoke({"input": user_input, "chat_history": memory.chat_memory.messages})

    # Extract the answer from the result
    answer = result.get("answer", "")

    # Append the user's message and the assistant's response to the chat history
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(answer)

    # Serialize and save the chat history in the session
    session['chat_history'] = serialize_messages(memory.chat_memory.messages)
    
    clean_answer = llm.invoke(output_cleanup_prompt.format(llm_response=answer))
    clean_answer = clean_answer.split("##Improved answer:##")[-1]

    return clean_answer

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )