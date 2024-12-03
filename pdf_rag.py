## 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user
## run pip install -r requirements.txt to install the required packages
# pip install --upgrade -r requirements.txt
# source venv/Scripts/activate

import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings  # Pour les embeddings

# Chemin vers le fichier PDF
doc_path = "./data/DOC.pdf"
model = "llama3.2"

print("Démarrage du traitement du PDF")

try:
    # Ouvrir le PDF
    pdf_document = fitz.open(doc_path)
    print("PDF chargé avec succès...")

    # Extraire le PDF et le convertir en objets Document de LangChain
    data = []
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text = page.get_text()
         # Créer un objet Document pour chaque page
        data.append(Document(page_content=text, metadata={"page_number": page_num}))

    # Lire la première page (juste pour l'aperçu)
    first_page = pdf_document[0]  # Index de la première page
    content = first_page.get_text()

    print("Aperçu de la première page :")
    print(content[:100])  # Afficher les 100 premiers caractères

    pdf_document.close()

except Exception as e:
    print(f"Une erreur est survenue : {e}")
    data = []

# from langchain_ollama import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma

# Split and chunk
if data:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(data)
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i}: {chunk.page_content[:100]}")
    print("done splitting....")

    print(f"Number of chunks: {len(chunks)}")
    print(f"Example chunk: {chunks[0]}")


# ===== Add to vector database ===
import ollama

ollama.pull("nomic-embed-text")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag",
)
print("done adding to vector database....")

## === Retrieval ===
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# set up our model to use
llm = ChatOllama(model=model)

# a simple technique to generate multiple questions from a single question and then retrieve documents
# based on those questions, getting the best of both worlds.
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of distance-based 
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}"""
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)
# --Test--

# print(retriever)

# question = "What are the benefits of Python?"
# generated_questions = QUERY_PROMPT.format(question=question)
# print(generated_questions)

# results = retriever.get_relevant_documents("Example question")
# for doc in results:
#     print(doc)
# --Test-End--

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
print(template)

prompt = ChatPromptTemplate.from_template(template)

print(prompt)
print(retriever)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# res = chain.invoke(input=("what is the document about?",))
# res = chain.invoke(
#     input=("what are the main points of the formation?",)
# )
res = chain.invoke(input=("how to report DOC?",))

print(res)
print("All ok")