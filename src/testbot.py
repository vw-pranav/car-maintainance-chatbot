from chatbot import ask_question

chat_history = []

while True:
    question = input("\nAsk: ").strip()

    if not question:
        continue

    if question.lower() == "exit":
        break

    chat_history.append({"role": "user", "content": question})
    result = ask_question(question, history=chat_history)
    answer = result["answer"]

    chat_history.append({"role": "assistant", "content": answer})

    print("\n================ ANSWER ================\n")
    print(answer)