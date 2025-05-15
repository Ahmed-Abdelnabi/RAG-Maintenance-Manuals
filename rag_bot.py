from langchain_groq import ChatGroq
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


class Rag_Bot:
    def __init__(self, api_key,
                 embedding_model = "all-MiniLM-L6-v2", 
                 num_results = 10,
                 num_top_reults = 10):
        self.api_key = api_key

        self.token_split_text = None
        self.vectorstore = None

        self.embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model)

        self.llm = ChatGroq(api_key=self.api_key, model="llama-3.3-70b-versatile",
                            temperature=0,max_tokens=None,timeout=None,max_retries=2)
        self.query = None

        self.prompt1 = ChatPromptTemplate.from_messages([
            ("system", 
            "You are a helpful expert Mechanical Engineer. Provide an example answer to the given question,\
            that might be found in an Installation, Operation, Maintenance and troubleshooting manufacturer manual regarding a {equipment_name}"),
            ("user", "{query}")
        ])   
        self.generated_answer = None
        self.num_results = num_results
        self.results = None
        self.num_top_results = num_top_reults
        self.prompt2 = ChatPromptTemplate.from_messages([
            ("system", 
            "You are a helpful expert Mechanical Engineer. Your users are asking questions about information contained in {equipment_name}'s\
            Installation, Operation, Maintenance or troubleshooting manufacturer manual.\n\nYou will be shown the user's question, and the\
            relevant information from the manufacturer manual. Answer the user's question ONLY USING these PROVIDED INFORMATION,\
            and if you think the provided information are not relevant just SAY that the answer doesn't exist in the Manual and don't\
            ever invent answers. Please always format your answer using Markdown. Use bullet points for lists, bold for emphasis, etc."),
            ("user", "Question: {query}. \n\nInformation:\n{information}")
        ]) 
        self.final_answer = None
    
    def tokenize_doc(self, doc_path):
        doc = PdfReader(doc_path)
        pdf_texts = [p.extract_text().strip() for p in doc.pages]
        # filter out empty strings if any
        pdf_texts = [text for text in pdf_texts if text]

        char_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len,
        )
        char_split_text = char_splitter.split_text("\n\n".join(pdf_texts))
        token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
        self.token_split_text = []
        for text in char_split_text:
            self.token_split_text += token_splitter.split_text(text)
        return self.token_split_text
    


    def create_faiss_index(self):
        if self.token_split_text:
            # Generate embeddings & create index
            self.vectorstore = FAISS.from_texts(
                texts = self.token_split_text,
                embedding = self.embedding_function)
            self.token_split_text = None # to remove the tokenized text after embedding
            message_2 = "Document Processed Successfully"
        else:
            message_2 = "Please upload a file first!! and try again"
        return message_2,self.vectorstore

    def generate_initial_answer(self, equipment_name):
        augmentation_chain = self.prompt1 | self.llm
        response = augmentation_chain.invoke(
            {"equipment_name": equipment_name,
            "query": self.query})
        self.generated_answer = response.content
        return self.generated_answer
    
    
    def query_index(self, vector_store):
        # Create the augmented query 
        augmented_query = self.query + "\n" + self.generated_answer

        # FAISS returns a list of results, where each result contains a 'document' and its 'score'
        results = vector_store.similarity_search(augmented_query, k=self.num_results)

        # Extract only unique documents
        unique_results = set()
        for res in results:
            unique_results.add(res.page_content)

        # Get only top n answers
        self.results = list(unique_results)[:self.num_top_results]
        return self.results
    
    def get_final_answer(self, equipment_name):
        final_chain = self.prompt2 | self.llm
        info = "\n\n".join(self.results)
        response = final_chain.invoke(
            {"equipment_name": equipment_name,
             "query": self.query,
             "information": info}
        )
        self.final_answer = response.content
        return self.final_answer
    
