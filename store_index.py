import os
from dotenv import load_dotenv
from logger import logging
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from src.helper import ConfigManager, generate_id
from src.data_processing import load_pdfs, split_data, process_data_chunks
from src.vectordb import discard_duplicated_chunks
from src.model import hf_load_embeddings

logging.info("Starting Pinecone vector DB data update process")

logging.info("Loading environment variables & config.json")
_ = load_dotenv()

config_manager = ConfigManager("config.json")

extracted_data = load_pdfs(config_manager.get_config("data_directory"), config_manager, verbose=True)
data_chunks = split_data(extracted_data, config_manager)
formatted_data_chunks = process_data_chunks(data_chunks)
logging.info(f"Number of data chunks: {len(formatted_data_chunks)}")

embeddings  =   hf_load_embeddings(config_manager)

pc = Pinecone(api_key=os.getenv("PINECONE_TOKEN"))
index = pc.Index(config_manager.get_config("pinecone_index_name"))

deduplicated_data_chunks = discard_duplicated_chunks(formatted_data_chunks, index, embeddings)
logging.info(f"Number of not duplicated data chunks: {len(deduplicated_data_chunks)}")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)
data_chunks_ids = [generate_id(chunk) for chunk in deduplicated_data_chunks]
logging.info(f"Loading data chunks into Pinecone vector database")
_ = vector_store.add_texts(texts=deduplicated_data_chunks, ids=data_chunks_ids)
logging.info(f"Data loaded into Pinecone vector database")