import os
from dotenv import load_dotenv

_ = load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true" # LangSmith configuration
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com" # LangSmith configuration
os.environ["LANGCHAIN_PROJECT"] = "medical-llm-using-llama3" # LangSmith configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # This can help mitigate memory fragmentation issues
