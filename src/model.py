import os
import torch
import transformers
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from src.helper import ConfigManager


def hf_download_model(config_manager: ConfigManager):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config_manager.get_config("llm_model_repository_name"), 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        token=os.getenv("HUGGINGFACE_TOKEN"),
    )

    _ = model.save_pretrained(config_manager.get_config("llm_model_directory"))

    return model

def hf_download_tokenizer(config_manager: ConfigManager):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config_manager.get_config("llm_model_repository_name"), 
        trust_remote_code=True,
        token=os.getenv("HUGGINGFACE_TOKEN")
    )

    _ = tokenizer.save_pretrained(config_manager.get_config("llm_model_directory"))

    return tokenizer

def load_local_model(config_manager: ConfigManager, tokenizer, verbose: bool = False):
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

    return llm

def hf_load_embeddings(config_manager: ConfigManager):
    embeddings = HuggingFaceEmbeddings(model_name=config_manager.get_config("embedding_model_name"))

    return embeddings