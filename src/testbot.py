from chatbot import ask_question

while True:

    question = input("\nAsk: ")

    if question.lower() == "exit":
        break

    result = ask_question(question)

    print("\n================ ANSWER ================\n")
    print(result["answer"])