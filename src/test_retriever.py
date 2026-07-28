from retrieval.retriever import get_retriever

retriever = get_retriever()

question = "How do I replace engine oil?"

results = retriever.invoke(question)

print("\nQuestion:")
print(question)

print("\nRetrieved Chunks:\n")

for i, doc in enumerate(results):

    print("=" * 80)

    print(f"Result {i+1}")

    print("\nMetadata:")
    print(doc.metadata)

    print("\nContent:\n")

    print(doc.page_content[:1000])