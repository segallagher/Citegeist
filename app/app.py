from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

from rag.models import embedding_model, llm_model
from rag.prompts import Prompts

from flask import Flask, render_template, request, jsonify
from markdown import markdown

# Global Vars
app = Flask(__name__)
chat_log: dict = []
load_dotenv(override=True)
vectorstore: Chroma = Chroma(
    persist_directory=os.environ["VECTORSTORE_DIR"],
    embedding_function=embedding_model(),
)

def add_chat_message(sender: str, message:str, context:list=[]) -> None:
    """
    Adds a message to a global chat_log.

    Parameters:
        sender (str): The type of message sender
        message (str): The contents of the message being sent.
        content (list, optional): documents the machine sent response used during generation
    
    Returns:
        None
    
    Side Effects:
        Updates global variable `chat_log` appending a new message.
    """
    chat_log.append(
        {
            "sender": sender,
            "message": message,
            "context": context,
        }
    )

def get_rendered_messages() -> list[dict]:
    """
    Formats global `chat_log` for rendering in webapp

    Returns:
        list[dict]: A list of message dictionaries formatted for display
    """
    return [
        {
            'sender': entry['sender'],
            'message': markdown(entry['message']),
            'context': [document.metadata for document in entry['context']],
        } for entry in chat_log
    ]

@app.route('/', methods=['GET', 'POST'])
def chat():
    render_params={
        "title": "Chat Log",
    }
    return render_template('chat.j2', **render_params)

def ask_llm(message:str) -> None:
    """
    Perform RAG and ask LLM the user's question

    Parameters:
        message (str): User question to ask the LLM
    
    Returns:
        None
    
    Side Effects:
        Adds machine response to global `chat_log`
    """
    retriever: VectorStoreRetriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(message)
    
    chain = Prompts.CITES_SOURCES | llm_model()
    response = chain.invoke({"context": docs, "question": message})
    add_chat_message(
        sender='machine',
        message=response.content,
        context=docs,
    )

@app.route('/send', methods=['POST'])
def send_message():
    # Get Data
    data = request.get_json()
    username = data.get('username')
    user_message = data.get('message')

    # Validate Data
    if not username:
        return jsonify({'error': 'No username provided'}), 400
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Append to chat history
    add_chat_message(
        sender=username,
        message=user_message,
    )

    # Get machine response
    ask_llm(user_message)

    return jsonify({'status': 'ok'})

@app.route('/messages', methods=["GET"])
def get_messages():
    return jsonify(get_rendered_messages())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
