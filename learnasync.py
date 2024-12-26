import asyncio
import PyPDF2
import chromadb
from chromadb.utils import embedding_functions
from groq import AsyncGroq

# Initialize clients
client = AsyncGroq(api_key="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")  
chroma_client = chromadb.Client()

# Create a collection in ChromaDB
collection_name = "pdf_chunks"
collection = chroma_client.create_collection(name=collection_name)

def read_and_chunk_pdf(pdf_path):
    extracted_lines = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: 
                extracted_lines.extend(page_text.split('\n'))

    full_text = '\n'.join(extracted_lines)

    def chunk_text(text, chunk_size=300):
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    return chunk_text(full_text, chunk_size=300)[4:]  # Skip first 4 chunks

def prepare_documents(chunks):
    documents = []
    metadatas = []
    ids = []
    
    for i, chunk in enumerate(chunks):
        documents.append(chunk) 
        metadata = {"source": "artificial_intelligence_tutorial.pdf", "chunk_index": i + 1}
        metadatas.append(metadata)  
        ids.append(f"chunk_{i + 1}") 

    return documents, metadatas, ids

async def retrieve_relevant_chunk(query, default_ef):
    query_embedding = default_ef(query)  

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=4 
    )
    
    if result['documents']:
        doc_id = result['ids'][0]  
        context_chunk = result['documents'][0]    
        return context_chunk, doc_id  
    else:
        return None, None  

async def request_groq(user_message, context):
    if context is None:
        context = "No relevant context available."

    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are an AI assistant. Your responses must be based only on the provided context below. Use this context: {context}."
            },
            {
                "role": "user",
                "content": user_message,
            }
        ],
        model="llama-3.1-70b-versatile",
        temperature=0.5,
        max_tokens=300,
        top_p=1,
        stop=None,
        stream=False,
    )

    return chat_completion.choices[0].message.content

async def main(queries, pdf_path):
    chunks = read_and_chunk_pdf(pdf_path)
    documents, metadatas, ids = prepare_documents(chunks)

    default_ef = embedding_functions.DefaultEmbeddingFunction()
    embeddings = default_ef(documents)

    # Add documents to the collection
    collection.add(
        documents=documents,  
        metadatas=metadatas, 
        ids=ids,
        embeddings=embeddings
    )

    tasks = []
    for query in queries:
        context_chunk, _ = await retrieve_relevant_chunk(query, default_ef)
        if context_chunk:  # Check if context_chunk is not None
            tasks.append(request_groq(query, context_chunk))  # Append the coroutine
        else:
            tasks.append(f"No relevant context for query: {query}")  # Handle the case with no context

    return await asyncio.gather(*tasks)

if __name__ == "__main__":
    queries = [
        "How to Bake a Cake?"
        "What is jio5g?"
        "Explain the importance of AI in modern technology.",
        "What are the goals of artificial intelligence?",
        "Describe the differences between supervised and unsupervised learning.",
        "What is reinforcement learning?",
        "Explain the concept of neural networks.",
        "What are the ethical implications of AI?",
        "How can AI be used in healthcare?",
        "What is natural language processing?",
        "Describe the Turing test.",
        "What are some common applications of computer vision?",
        "How does machine learning differ from traditional programming?",
        "What is the role of data in machine learning?",
        "What are some popular machine learning frameworks?",
        "Explain the concept of overfitting in machine learning.",
        "What is cross-validation?",
        "How do decision trees work?",
        "What are support vector machines?",
        "What is deep learning?",
        "Explain the architecture of a convolutional neural network.",
        "What are generative adversarial networks (GANs)?",
        "How is AI impacting the job market?",
        "What are the different types of machine learning algorithms?",
        "How do you evaluate the performance of a machine learning model?",
        "What is feature engineering?",
        "What is clustering in machine learning?",
        "Explain the importance of feature selection.",
        "What are some challenges in deploying AI models?",
        "How does transfer learning work?",
        "What is the role of the AI ethics board?",
        "How can bias be reduced in AI models?",
        "What is the future of AI in education?",
        "Explain the concept of explainable AI.",
        "What are some recent breakthroughs in AI research?",
        "How do recommender systems work?",
        "What is the difference between AI, machine learning, and deep learning?",
        "What is the significance of the AI winter?",
        "How do robots use AI for navigation?",
        "What are the main components of an AI system?",
        "What is autonomous driving, and how does AI play a role?",
        "Explain the concept of reinforcement learning in gaming.",
        "What are the benefits of using AI in marketing?",
        "How does AI assist in fraud detection?",
        "What are some applications of AI in finance?",
        "What is the role of AI in climate change?",
        "How does AI contribute to smart cities?",
        "What are the risks associated with AI?",
        "What is the importance of transparency in AI?",
        "Explain the concept of agent-based modeling in AI."
    ]

    pdf_path = 'artificial_intelligence_tutorial.pdf'  # Specify your PDF path here
    parallel_response = asyncio.run(main(queries=queries[:25], pdf_path=pdf_path))
    
    for response in parallel_response:
        print(response, end="\n" * 5)
