from flask import Flask, render_template, request
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

app = Flask(__name__)
load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# ✅ Use the correct Gemini model name
chatModel = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# Prompt template (optional, if you want structured messages)
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
    print(f"User: {msg}")

    # ✅ Proper message format
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=msg)
    ]

    response = chatModel.invoke(messages)

    # ✅ Only return the text response
    return response.content

@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
