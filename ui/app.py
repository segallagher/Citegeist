from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
import os
from dotenv import load_dotenv

from util.models import embedding_model, llm_model
from util.prompts import Prompts

from flask import Flask, render_template, request, redirect, url_for, jsonify
from markdown import markdown

# Global Vars
app = Flask(__name__)
chat_log: dict = []
load_dotenv(override=True)
vectorstore: Chroma = Chroma(
    persist_directory=os.environ["VECTORSTORE_DIR"],
    embedding_function=embedding_model(),
)

def add_chat_message(username: str, message:str):
    """
    Adds a message to a global chat_log
    """
    chat_log.append(
        {
            "sender": username,
            "message": message,
        }
    )

@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        username = request.form['username']
        user_message = request.form['message']

        chat_log.append({'sender': username, 'message': user_message})
        return redirect(url_for('chat'))

    render_params={
        "title": "Chat Log",
    }
    return render_template('chat.j2', **render_params)

def ask_llm(message:str) -> None:
    """
    Perform RAG and ask LLM the user's question
    """
    retriever: VectorStoreRetriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(message)
    
    chain = Prompts.CITES_SOURCES | llm_model()
    response = chain.invoke({"context": docs, "question": message})
    add_chat_message(username='machine', message=response.content)

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
    chat_log.append({'sender': username, 'message': user_message})

    # Get machine response
    ask_llm(user_message)

    return jsonify({'status': 'ok'})

@app.route('/messages', methods=["GET"])
def get_messages():
    rendered_chat_log = [
        {
            'sender': entry['sender'],
            'message': markdown(entry['message'])
        } for entry in chat_log
    ]
    return jsonify(rendered_chat_log)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
