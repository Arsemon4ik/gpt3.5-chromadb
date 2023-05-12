from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.vectorstores import Chroma


def main():
    load_dotenv()

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("You are not provide the OPENAI_API_KEY")
        exit(1)

    # temperature for uniqueness ai answers
    # chat model gpt-3.5-turbo set by default
    chat = ChatOpenAI(temperature=0.8)

    messages = [
        SystemMessage(content="You are a helpful assistant"),
    ]

    print("Hi! I'm AI assistant CLI. To exit type CTRL+C")

    while True:
        try:
            user_input = input(":-> ")

            messages.append(HumanMessage(content=user_input))

            ai_response = chat(messages)

            messages.append(AIMessage(content=ai_response.content))

            print("Answer:", ai_response.content)
        except KeyboardInterrupt:
            print("The Program is terminated!")
            raise SystemExit


if __name__ == "__main__":
    main()