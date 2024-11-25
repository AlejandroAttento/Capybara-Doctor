import os
import torch
import logging
import transformers
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from src.helper import ConfigManager


def hf_download_model(config_manager: ConfigManager):
    try:
        logging.basicConfig(level=logging.INFO)
        logging.info("Starting model download...")

        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN is not set.")

        repository_name = config_manager.get_config("llm_model_repository_name")
        model_directory = config_manager.get_config("llm_model_directory")

        logging.info(f"Repository Name: {repository_name}")
        logging.info(f"Model Directory: {model_directory}")

        model = transformers.AutoModelForCausalLM.from_pretrained(
            repository_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            token=token,
        )

        logging.info("Model downloaded successfully.")
        model.save_pretrained(model_directory)
        logging.info(f"Model saved to disk {model_directory}.")

        return model

    except Exception as e:
        logging.error(f"An error occurred while downloading the model: {e}")
        raise

def hf_download_tokenizer(config_manager: ConfigManager):
    try:
        logging.basicConfig(level=logging.INFO)
        logging.info("Starting tokenizer download...")

        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN is not set.")


        repository_name = config_manager.get_config("llm_model_repository_name")
        model_directory = config_manager.get_config("llm_model_directory")

        logging.info(f"Repository Name: {repository_name}")
        logging.info(f"Model Directory: {model_directory}")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            repository_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            token=token,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer
        
    except Exception as e:
        logging.error(f"An error occurred while downloading the model: {e}")
        raise

def load_local_model(config_manager: ConfigManager, tokenizer, verbose: bool = False):
    logging.info("Starting local model loading...")

    generation_config = transformers.GenerationConfig(
        max_length=config_manager.get_config(["model_params", "max_length"]),           
        max_new_tokens=config_manager.get_config(["model_params", "max_new_tokens"]),  
        temperature=config_manager.get_config(["model_params", "temperature"]),
        repetition_penalty=config_manager.get_config(["model_params", "repetition_penalty"]),
        no_repeat_ngram_size=config_manager.get_config(["model_params", "no_repeat_ngram_size"]),
        early_stopping=bool(config_manager.get_config(["model_params", "early_stopping"])),
        do_sample=bool(config_manager.get_config(["model_params", "do_sample"])),
        num_beams=config_manager.get_config(["model_params", "num_beams"])
    )

    logging.info(f"Model configuration :{generation_config}")

    try:
        pipeline = transformers.pipeline(
            "text-generation",
            model=config_manager.get_config("llm_model_directory"),
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            generation_config=generation_config,
        )

        llm = HuggingFacePipeline(pipeline=pipeline, verbose=verbose)
        logging.info("Model loaded")

        return llm
    
    except Exception as e:
        logging.error(f"An error occurred while downloading the model: {e}")
        raise

def hf_load_embeddings(config_manager: ConfigManager):
    try:
        logging.info("Starting embeddings model downloading and loading...")
        embeddings = HuggingFaceEmbeddings(model_name=config_manager.get_config("embedding_model_name"))
        logging.info("Embeddings model loaded")

        return embeddings

    except Exception as e:
        logging.error(f"An error occurred while downloading and loading the embeddings model: {e}")
        raise

    