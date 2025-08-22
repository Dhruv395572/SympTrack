from flask import Flask, render_template, request
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

app = Flask(__name__)
load_dotenv()

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

chatModel = ChatGroq(model="llama-3-8b", api_key=GROQ_API_KEY)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(msg)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": msg}
    ]
    response = chatModel.invoke(messages)
    return str(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)