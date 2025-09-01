import json
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load the knowledge base from JSON
with open("knowledge_base.json", "r") as f:
    knowledge_base = json.load(f)

# Create documents for the vector store
documents = [
    Document(
        page_content=f"Question: {item['question']}\nAnswer: {item['answer']}",
        metadata={"question": item['question']}
    ) for item in knowledge_base
]

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Create FAISS vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# Initialize the LLM with faster model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# Simple retriever with reduced k for speed
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Custom prompt template
prompt_template = """
You are a helpful customer support chatbot for ZenexCloud hosting services.
Use the following context to answer the user's question accurately and concisely.
If the question is not related to the knowledge base, politely say "You need a human technician for this, please provide us your email and we will contact you shortly."

Context:
{context}

User Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Chatbot loop
def chatbot():
    print("Welcome to ZenexCloud Customer Support Chatbot! Type 'exit' to quit.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        response = qa_chain.invoke({"query": user_query})
        answer = response['result']
        print(f"Chatbot: {answer}")

if __name__ == "__main__":
    chatbot()