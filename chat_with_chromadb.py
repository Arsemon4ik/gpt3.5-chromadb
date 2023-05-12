from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

TEMPLATE = """
You are a chatbot having a conversation with a human.
This is chat history for context. You may use it if you need. But it is not necessary.
{chat_history}
This is current chat with this user:
Human: {human_input}
Chatbot:"""


class Chat:
    """Class for saving the context to the Chroma db while chatting with GPT"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        # if you need a memory "on the go"
        # self.memory = ConversationBufferMemory(memory_key='chat_history')
        self.prompt = PromptTemplate(input_variables=["chat_history", "human_input"],
                                     template=TEMPLATE)

        self.llm_chain = LLMChain(
            llm=OpenAI(),
            prompt=self.prompt,
            verbose=True
        )
        # if you want to save memory locally via the Chroma
        self.vectordb = self._get_vector()

    def _get_vector(self):
        """Instruction for finding the Chroma db locally or create the new one."""
        if os.path.isdir('db'):
            vectordb = Chroma(persist_directory='db',
                              embedding_function=self.embeddings)
        else:
            text_splitter = RecursiveCharacterTextSplitter()
            start_chunk = text_splitter.split_text("You are a helpful assistant")
            vectordb = Chroma.from_texts(start_chunk,
                                         self.embeddings,
                                         persist_directory='db')
            vectordb.persist()
        return vectordb

    def _get_all_context(self) -> str:
        """Instruction for getting all the context from Chroma db."""
        return "\n".join([text for text in self.vectordb.get()['documents']])

    def _get_relevant_documents(self, query: str) -> str:
        """Instruction for getting all the relevant context from Chroma db using get_relevant_documents"""
        retriever = self.vectordb.as_retriever(search_type="mmr")
        docs = retriever.get_relevant_documents(query)
        return "\n".join([document.page_content for document in docs])

    def _get_similarity_context(self, query: str, number_docs: int = 2) -> str:
        """Instruction for getting all the relevant context from Chroma db using similarity_search"""
        return "\n".join(
            [document.page_content for document in self.vectordb.similarity_search(query=query, k=number_docs)])

    def ask(self, query: str) -> str:
        """Instruction for asking the GPT model while adding the new context to the db"""
        user_input = query.lower()

        # You can change the searching method if you want
        docs2 = self._get_relevant_documents(query=user_input)
        response = self.llm_chain.run(chat_history=docs2, human_input=user_input)
        self.vectordb.add_texts(["Human: " + user_input, "Chatbot:" + response])
        return response


def main():
    st.set_page_config('Talk to the chat-GPT')
    st.header('Ask a question 💬')
    user_input = st.text_input('Write a question: ')
    if user_input:
        chat = Chat()
        ai_response = chat.ask(user_input)
        st.write(ai_response)


if __name__ == "__main__":
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("You are not provide the OPENAI_API_KEY")
        exit(1)

    main()
