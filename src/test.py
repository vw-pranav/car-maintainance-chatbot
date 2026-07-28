from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma-local:latest")

for chunk in llm.stream("What is a car engine?"):
    print(chunk.content, end="", flush=True)