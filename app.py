from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt

chatModel = ChatGroq(model="llama-3-8b-8192", api_key=GROQ_API_KEY)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        print(msg)
        # List of BaseMessages pass karo
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": msg}
        ]
        response = chatModel.invoke(messages)
        # Agar response dict hai aur "content" key hai to wahi return karo
        if isinstance(response, dict) and "content" in response:
            return str(response["content"])
        return str(response)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)