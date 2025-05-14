#!/usr/bin/env python
# coding: utf-8

# # AI-Powered Legal Document Summarizer Built with Gemini & RAG
# ## ⚖️ Legalight: Your AI Ally for Safer Agreements  
# **Know Before You Agree — Because Fine Print Shouldn't Be a Trap**
# 
# ## 🚀 Use Case 
# Legal documents are everywhere, from app Terms & Conditions to employment contracts and lease agreements. Yet most people **never read them** due to their **length**, **complexity**, and **legal jargon**. This leads to:
# 
# - 📜 Accidental agreement to hidden terms  
# - 💸 Unexpected penalties and auto-renewals  
# - 🔒 Unaware data-sharing or privacy breaches  
# 
# **Legalight** solves this by using **Generative AI** to help users **understand, summarize, and question** any legal document in seconds — directly within this notebook.
# 
# 💡 **What makes it unique?**  
# Instead of static document analysis, Legalight turns legal content into a **dynamic conversation** using:
# 
# - Structured outputs (JSON Mode)
# - Document summarization & clause extraction
# - Risk level classification
# - Vector search & embeddings for accurate responses
# 
# This AI-powered assistant makes legal language **understandable for everyone**, from individuals to small business owners.
# 
# ## ⚖️ Blog Post Link 
# 
# Medium Link: https://medium.com/@sanaishfaq25/legalight-simplifying-legal-documents-with-generative-ai-0ef1b2b00ee8
# 
# ## 📄 Project Overview  
# Legalight is a generative AI-powered legal assistant that:
# 
# 1. Accepts any uploaded `.txt` or `.pdf` document  
# 2. Analyzes and summarizes the entire content  
# 3. Extracts and labels important clauses with risk levels
# 
# ## 📊 GenAI Capabilities Used
# 
# | Capability                                | Description                                                                 |
# |-------------------------------------------|-----------------------------------------------------------------------------|
# | 📑 Document Understanding                | Analyzes and interprets legal documents, extracts key clauses, and generates summaries. |
# | 🧠 Structured Output (JSON Mode)          | Organizes outputs (e.g., clauses and risk levels) in a structured format for easy interpretation. |
# | 🔎 Retrieval-Augmented Generation (RAG)   | Retrieves relevant document sections and enhances model responses with contextual data. |
# | 📚 Embeddings & Vector Search            | Converts text into meaningful vector representations for semantic search and content retrieval. |
# 
# 
# ## ❗ Why This Problem Matters  
# In the real world, most people accept legal terms without reading them — whether it's for using a new app, signing a rental agreement, or entering a service contract. The language is often **overwhelming**, **time-consuming**, and **intentionally confusing**. As a result, users unknowingly agree to:
# 
# - Unwanted auto-renewals  
# - Restrictive refund or cancellation policies  
# - Loss of rights or data privacy  
# 
# Legalight empowers users to **take control**, **understand what they agree to**, and make informed decisions — instantly and interactively.
# 
# ## 🌟 Benefits to Users
# 
# - ⏱️ Save time by skipping lengthy reading  
# - 🛡️ Avoid risky legal commitments or hidden fees  
# - 🤝 Improve trust in digital agreements  
# - 📚 Enhance awareness of rights and obligations  
# - 💼 Ideal for individuals, freelancers, and small business owners
# 
# 

# ## 🔧 Step 1: Install Required Libraries
# 
# We’ll start by installing the essential libraries required for this project. These include tools to interact with the Gemini API, handle document processing, enable retrieval-augmented generation (RAG), and structure our AI outputs. This setup ensures that **Legalight** can analyze documents, extract meaningful insights, and respond intelligently.

# In[1]:


# Install necessary libraries
get_ipython().system('pip install -q google-generativeai langchain langchain-community langchain_google_genai langchain_chroma pydantiC')


# ## 🧠 Step 2: Import Libraries and Modules
# 
# Next, we import all the required modules and tools to build the functionality of **Legalight**. These include:
# 
# - File loaders for PDFs and text files.
# - Tools for splitting long documents into manageable chunks.
# - Google Gemini-based models for both text generation and embeddings.
# - Chroma for storing and retrieving vector-based document chunks.
# - Pydantic for defining structured output models.
# - Kaggle secrets to securely access the Gemini API key.
# 
# These imports set the foundation for document processing, intelligent querying, and structured AI outputs.

# In[2]:


import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from typing import List, Optional
from kaggle_secrets import UserSecretsClient


# ## 🔑 Step 3: Configure Gemini API Key
# 
# In this step, we configure the **Google Gemini API** by securely fetching the API key from Kaggle secrets. The key is essential for accessing the AI models and performing the document analysis.
# 
# If the key is not found in the environment or Kaggle secrets, an error will be raised, ensuring that the setup is complete before proceeding.
# 
# Once the API key is successfully retrieved, the environment is ready for use.

# In[3]:


# Configure Gemini API Key
try:
    user_secrets = UserSecretsClient()
    GEMINI_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
except:
    raise ValueError("Please set GOOGLE_API_KEY in Kaggle Secrets or environment.")
print("Environment set up successfully!")


# ## 📄 Step 4: Load Document
# 
# This function is responsible for loading the uploaded document, whether it is in **.pdf** or **.txt** format. The document is parsed and prepared for analysis. 
# 
# - First, it checks the file extension to determine whether the document is a PDF or a text file.
# - If the file is a **PDF**, it uses the `PyPDFLoader` to load the document.
# - If the file is a **text file**, it uses the `TextLoader` to load the content.
# - If the file format is unsupported, an error message is raised, prompting the user to upload a valid document.
# 
# Once the document is loaded successfully, it returns the parsed pages for further processing.

# In[4]:


def load_document(filepath):
    """Loads .txt or .pdf document and returns parsed pages."""
    print("Loading document...", end=' ')
    file, extension = os.path.splitext(filepath)
    if extension == '.pdf':
        loader = PyPDFLoader(filepath)
    elif extension == '.txt':
        loader = TextLoader(filepath)
    else:
        raise ValueError("❌ Unsupported file format. Please upload a .pdf or .txt file.")
    print("✅ Done")
    return loader.load()


# ## 🔍 Step 5: Create Vector Store for Semantic Search
# 
# This function creates a vector store, which is essential for performing semantic search on the legal document chunks. Here's how it works:
# 
# - The function uses the **GoogleGenerativeAIEmbeddings** model to convert the document chunks into vector embeddings, which are numerical representations of the text.
# - The **Chroma** library is then used to create the vector store, where the document chunks are stored and indexed. This store enables fast and efficient similarity searches.
# - The vector store is named **TrustTermsAI** to represent the project and is configured with the **Google Gemini API** embeddings.
# 
# Once the vector store is ready, it allows the system to retrieve relevant chunks of text during question-answering and summarization tasks.

# In[5]:


def create_vector_store(chunks):
    print("📦 Creating vector store for semantic search...")
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GEMINI_API_KEY)
    vector_store = Chroma.from_documents(collection_name="Legalight", embedding=embedding_model, documents=chunks)
    print("✅ Vector store ready!")


# ## 🔄 Step 6: Retrieve Relevant Document Chunks
# 
# In this step, we define the **retrieve** function, which performs a semantic search to fetch the most relevant document chunks based on a user's question. Here's the process:
# 
# - **Embedding Model:** The function utilizes the **GoogleGenerativeAIEmbeddings** model to generate embeddings for the question being asked.
# - **Vector Database:** The **Chroma** vector store (created earlier) is used to retrieve the most relevant document chunks that are semantically similar to the question.
# - **Retriever:** The **Chroma** vector store is set up as a retriever, which searches for the top 3 most relevant document chunks based on the question using **k=3** in the search arguments.
# 
# This function ensures that the chatbot responds with the most relevant content from the legal document to the user's query.

# In[6]:


def retrieve(question):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GEMINI_API_KEY)
    vector_db = Chroma(collection_name="Legalight", embedding_function=embedding_model)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    return "\n".join([doc.page_content for doc in docs])


# ## 📝 Step 7: Define Structured Output for Summary and Clauses
# 
# Here, we define the structure for the output that will contain the **summary** of the legal document as well as the **key clauses** identified, along with their **risk assessments**. This is done using **Pydantic models** to ensure proper structure and data validation.
# 
# - **Clause Model:** Represents a single legal clause, including the **content** of the clause and its **risk level** (Low, Medium, or High).
# - **Summary Model:** Represents the overall **summary** of the document and a list of **important clauses**, each with an associated risk assessment.
# 
# These models ensure that the information is organized and can be easily used to communicate the findings from the document analysis.

# In[7]:


# Define the structure of the summary and key clauses output
class Clause(BaseModel):
    """
    Structured representation of a legal clause with risk assessment.
    """
    content: str = Field(description="Clause content")
    risk_level: str = Field(description="Risk level: Low, Medium, or High")

class Summary(BaseModel):
    """
    Structured representation of document summary and clauses.
    """
    summary: str = Field(description="Summary of the document")
    clauses: List[Clause] = Field(description="List of important clauses with risk assessment")


# ## 🛠️ Step 8: Load Dataset and Run Analysis
# 
# In this step, we are downloading the latest dataset using **KaggleHub** and processing the uploaded legal document for analysis.
# 
# - **Dataset Download:** The dataset `legal-terms` is downloaded, and we specify the path where it's located.
# - **Document Loading:** We load the default document (`terms.txt`), but you can upload your own document to analyze by placing it in the `agreesafeai-data` folder and adjusting the file path accordingly.
# - **Text Processing:** We concatenate the document text and split it into smaller chunks for efficient processing.
# - **Vector Store Creation:** The document chunks are used to create a **vector store** for semantic search, which helps in analyzing the document efficiently.
# - **Legal Document Summarization:** We use the **Google Gemini 2.0 Flash model** to analyze and summarize the document. The model extracts important clauses and evaluates the risk level associated with each clause.
# 
# ### Key Actions:
# - Load and process legal document
# - Split document into manageable chunks
# - Create vector store for semantic search
# - Analyze document with Google Gemini AI
# - Generate summary and key clauses with risk levels
# 
# This is a crucial step to extract valuable insights from the document and make it easier to understand key legal points and risks.

# In[8]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("sanaishfaq25/legal-terms")

print("Path to dataset files:", path)

# Upload a legal document (example .txt file)
legal_document_input="/kaggle/input/legal-terms/terms.txt"
documents = load_document(legal_document_input)
print("Using the default input document... If you want to use another document, please upload it in 'Data' section under the 'agreesafeai-data' folder with file named as input.txt or input.pdf, and replace the 'legal_document_input' variable in the code with new file path.")

# Combine all text into one string for summarization
complete_document_text = "\n".join([page.page_content for page in documents])

# Split long documents into manageable chunks for embeddings
print("🔍 Splitting document into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", ".", "\n", " ", ""],
    chunk_size=450
)
chunks = text_splitter.split_documents(documents=documents)
create_vector_store(chunks)
print(f"🧩 Created {len(chunks)} chunks.")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)

summarize_llm = llm.with_structured_output(Summary)

# Define prompt for summarization
summarize_prompt = [
    ("system", "You are a legal AI assistant that can understand the legal documents"),
    ("human", f"Analyze the following text from a legal document, TEXT: {complete_document_text}. Provide a short summary and Extract key clauses. For each clause, specify - Content of the clause, Risk level (Low, Medium, or High).")
]
# Run Analysis - Summarization & Key Clauses Extraction
print("🧠 Analyzing and summarizing the document using gemini-2.0-flash model...")
response = summarize_llm.invoke(summarize_prompt)
print("✅ Analysis complete!")

print("\n📋 Document Summary:\n")
print(response.summary)
print("\n🧾 Key Clauses:\n")
for idx, clause in enumerate(response.clauses):
    print(f"🔹 Clause {idx+1}: {clause.content}\n⚠️ Risk Level: {clause.risk_level}\n")

print("\n 🎉 You’re now legally enlightened! Thanks for exploring Legalight.")


# ## 🎉 Conclusion: Legalight - Your Personal Legal Assistant
# 
# Legalight is here to make dealing with legal documents easier and less overwhelming. By using cutting-edge **Generative AI** technology like **Google Gemini**, we've created a tool that:
# 
# - **Simplifies Complex Legal Language:** Legalight quickly summarizes long and confusing legal text, highlighting the most important parts so you don't have to dig through the fine print.
# - **Automatically Identifies Risks:** It helps you spot risky clauses in legal agreements, like hidden fees or penalties, so you can make informed decisions before you agree to anything.
# - **Makes Legal Stuff Accessible:** Whether you're a business owner, student, or just someone trying to understand a document, Legalight is designed to break down legal jargon and make it easier for everyone to understand their rights and responsibilities.
# 
# Looking ahead, we see great potential for Legalight to handle even more types of legal documents, offer deeper insights, and become an essential tool for anyone navigating the complex world of contracts and agreements.
# 
# Thanks for exploring **Legalight**now you're one step closer to being legally enlightened! 🧠⚖️
