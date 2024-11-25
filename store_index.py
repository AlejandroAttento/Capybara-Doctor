import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import ConfigManager, generate_id
from src.data_processing import load_pdfs, split_data, process_data_chunks
from src.vectordb import discard_duplicated_chunks
from src.model import hf_load_embeddings
from logger import logger
import argparse

def create_index():
    logger.info(f"Creating index {config_manager.get_config(['pinecone', 'index_name'])}...")
    pc.create_index(
        name=config_manager.get_config(["pinecone", "index_name"]),
        dimension=config_manager.get_config(["pinecone", "dimensions"]),
        metric=config_manager.get_config(["pinecone", "metric"]),
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled"
    )
    logger.info(f"Index {config_manager.get_config(['pinecone', 'index_name'])} created")  

def update_index(deduplicate: bool = False):
    logger.info(f"Updating index {config_manager.get_config(['pinecone', 'index_name'])}...")
    index = pc.Index(config_manager.get_config(["pinecone", "index_name"]))

    logger.info(f"Processing PFDs in {config_manager.get_config('data_directory')}...")
    extracted_data = load_pdfs(config_manager.get_config("data_directory"), config_manager, verbose=True)
    data_chunks = split_data(extracted_data, config_manager)
    formatted_data_chunks = process_data_chunks(data_chunks)
    logger.info(f"Number of data chunks created: {len(formatted_data_chunks)}")

    embeddings = hf_load_embeddings(config_manager)

    if deduplicate:
        logger.info(f"Deduplicating data chunks against existing ones in Pinecone vector database...")
        deduplicated_data_chunks = discard_duplicated_chunks(formatted_data_chunks, index, embeddings)
        logger.info(f"Number of not duplicated data chunks: {len(deduplicated_data_chunks)}")
    else:
        deduplicated_data_chunks = formatted_data_chunks

    if len(deduplicated_data_chunks) == 0:
        logger.info(f"No data will be added to Pinecone vector database")
    else:
        logger.info(f"Loading data chunks into Pinecone vector database")
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        data_chunks_ids = [generate_id(chunk) for chunk in deduplicated_data_chunks]
        _ = vector_store.add_texts(texts=deduplicated_data_chunks, ids=data_chunks_ids)
        logger.info(f"Data loaded into Pinecone vector database")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage Pinecone vector DB processes")
    parser.add_argument("command", nargs="?", choices=["regenerate"], help="Specify an action to perform")
    parser.add_argument("-deduplicate", action="store_true", help="Trigger deduplication during the update process")

    args = parser.parse_args()

    logger.info("Loading environment variables & config.json")
    _ = load_dotenv()
    config_manager = ConfigManager("config.json")

    logger.info("Setting up connection with Pinecone vector DB")
    pc = Pinecone(api_key=os.getenv("PINECONE_TOKEN"))

    if not args.command:
        logger.info("Starting Pinecone vector DB data update process")
        if config_manager.get_config(["pinecone", "index_name"]) not in pc.list_indexes().names():
            logger.info(f"Index {config_manager.get_config(['pinecone', 'index_name'])} does not exist.")
            create_index()

        update_index(args.deduplicate)

    elif args.command == "regenerate":
        logger.info("Starting Pinecone vector DB data regeneration process")
        logger.info(f"Deleting index {config_manager.get_config(['pinecone', 'index_name'])}")
        pc.delete_index(config_manager.get_config(["pinecone", "index_name"]))

        create_index()

        update_index(False)
    else:
        raise ValueError(f"Invalid command: {args.command}")
