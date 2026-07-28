import pandas as pd

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper



df = pd.read_excel("data/evaluation.xlsx")

dataset = Dataset.from_dict(
    {
        "question": df["Question"].tolist(),
        "answer": df["Actual Answer"].tolist(),
        "contexts": df["Retrieved Context"].apply(lambda x: [x]).tolist(),
        "ground_truth": df["Expected Answer"].tolist(),
    }
)



llm = ChatOllama(
    model="gemma-local:latest",
    temperature=0
)

ragas_llm = LangchainLLMWrapper(llm)



embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)


result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)



print("\n==============================")
print("RAGAS Evaluation Result")
print("==============================\n")

print(result)

print("\n==============================")
print(result.to_pandas())
print("==============================")