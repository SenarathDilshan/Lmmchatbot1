import gradio as gr
from langchain_community.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize HuggingFaceHub LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature": 0.1, "max_length": 500},
    huggingfacehub_api_token="<your_huggingface_api_token>"  # Replace with your token
)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Load and process PDF document
loader = PyPDFLoader("Senarath.pdf")  # Ensure this file is uploaded to your Space
docs = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Create retrievers
vectorstore = Chroma.from_documents(chunks, embedding_model)
vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

keyword_retriever = BM25Retriever.from_documents(chunks)
keyword_retriever.k = 2

# Combine retrievers into an ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[vectorstore_retriever, keyword_retriever],
    weights=[0.5, 0.5]
)

# Define prompt template
template = """
Answer this question using the provided context only.

{question}

Context:
{context}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Define the chain
chain = (
    {"context": ensemble_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Define the Gradio interface
def chatbot(input_text):
    try:
        response = chain.invoke(input_text)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(
    fn=chatbot,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text",
    title="RAG Chatbot",
    description="A Retrieval-Augmented Generation chatbot powered by LangChain and Hugging Face."
)

if __name__ == "__main__":
    iface.launch()
