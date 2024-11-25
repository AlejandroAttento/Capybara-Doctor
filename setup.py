from setuptools import find_packages, setup

setup(
    name='Capybara Doctor',
    version='1.0.0',
    author="Alejandro Daniel Attento",
    author_email="alejandro.attento@gmail.com",
    packages=find_packages(),
    install_requires=[
        "accelerate==1.1.1",
        "torch==2.5.1",
        "transformers==4.46.3",
        "langchain==0.3.8",
        "langchain-community==0.3.8",
        "langchain-pinecone",
        "langchain-huggingface",
        "flask==3.1.0",
        "flask-session==0.8.0",
        "pinecone==5.3.1",
        "pypdf==5.1.0",
    ]
)