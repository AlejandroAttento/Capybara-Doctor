from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from src.helper import ConfigManager


def load_pdfs(directory_path: str, config_manager: ConfigManager, glob: str = "*.pdf", verbose: bool = False):
    loader = DirectoryLoader(
        config_manager.get_config("data_directory"),
        glob=glob,
        loader_cls=PyPDFLoader,
        show_progress=verbose
    )

    extracted_data = loader.load()

    return extracted_data

def split_data(extracted_data, config_manager: ConfigManager):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config_manager.get_config(["text_split", "chunk_size"]), 
        chunk_overlap=config_manager.get_config(["text_split", "chunk_overlap"])
    )

    data_chunks = text_splitter.split_documents(extracted_data)

    return data_chunks

def process_data_chunks(data_chunks):
    content_data_chunks = [chunk.page_content for chunk in data_chunks]

    formatted_data_chunks = []
    for text in content_data_chunks:
        cleaned_text = ' '.join(text.split())
        cleaned_text = cleaned_text.replace('•', '-').replace(';', ',')
        formatted_data_chunks.append(cleaned_text)

    return formatted_data_chunks