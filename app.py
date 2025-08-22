from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt

chatModel = ChatGroq(model="llama-3-8b-8192", api_key=GROQ_API_KEY)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)

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
        from langchain_openai import OpenAIEmbeddings
        from langchain_pinecone import PineconeVectorStore

        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        index_name = "medical-chatbot"
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        msg = request.form["msg"]
        input = msg
        print(input)
        response = rag_chain.invoke({"input": msg})
        return str(response["answer"])
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)