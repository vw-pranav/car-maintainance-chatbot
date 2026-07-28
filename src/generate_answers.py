import pandas as pd
from chatbot import ask_question

excel_path = "data/evaluation.xlsx"

df = pd.read_excel(excel_path)

actual_answers = []
contexts = []

for question in df["Question"]:

    result = ask_question(question)

    actual_answers.append(result["answer"])
    contexts.append(result["context"])

df["Actual Answer"] = actual_answers
df["Retrieved Context"] = contexts

df.to_excel(excel_path, index=False)

print("Evaluation file updated successfully.")