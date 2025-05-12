import os
from dotenv import load_dotenv
import openai
import numpy as np

from langchain_groq import ChatGroq

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

from chromadb import EphemeralClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from langchain_core.prompts import ChatPromptTemplate

import streamlit as st


class Rag_Bot:
    def __init__(self, api_key,
                 embedding_model = "all-MiniLM-L6-v2", 
                 num_results = 10,
                 num_top_reults = 10):
        self.api_key = api_key

        self.token_split_text = None
        self.client = EphemeralClient()
        self.collections = [collection.name for collection in self.client.list_collections()]
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name= embedding_model,
                                                                       device= 'cpu')
        self.llm = ChatGroq(api_key=self.api_key, model="llama-3.3-70b-versatile",
                            temperature=0,max_tokens=None,timeout=None,max_retries=2)
        self.query = None
        self.selected_collection = None
        self.prompt1 = ChatPromptTemplate.from_messages([
            ("system", 
            "You are a helpful expert Mechanical Engineer. Provide an example answer to the given question,\
            that might be found in an Installation, Operation, Maintenance and troubleshooting manufacturer manual regarding a {collection}"),
            ("user", "{query}")
        ])   
        self.generated_answer = None
        self.num_results = num_results
        self.results = None
        self.num_top_results = num_top_reults
        self.prompt2 = ChatPromptTemplate.from_messages([
            ("system", 
            "You are a helpful expert Mechanical Engineer. Your users are asking questions about information contained in {collection}'s\
            Installation, Operation, Maintenance or troubleshooting manufacturer manual.\n\nYou will be shown the user's question, and the\
            relevant information from the manufacturer manual. Answer the user's question USING ONLY these PROVIDED INFORMATION,\
            and if you think the provided information are not relevant just SAY that the answer doesn't exist in the Manual and don't\
            ever invent answers. Please always format your answer using Markdown. Use bullet points for lists, bold for emphasis, etc."),
            ("user", "Question: {query}. \n\nInformation:\n{information}")
        ]) 
        self.final_answer = None
    
    def tokenize_doc(self, doc_path):
        doc = PdfReader(doc_path)
        pdf_texts = [p.extract_text().strip() for p in doc.pages]
        # filter out empty strings
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
    
    def create_get_collection(self, collection_name):
        self.client = EphemeralClient()
        if self.token_split_text:
            collection_name = collection_name.replace(" ", "_").title()
            collection = self.client.create_collection(name=collection_name, 
                                                       embedding_function=self.embedding_function,
                                                       get_or_create=True)
            ids_list = collection.get().get('ids', [])
        
            if ids_list:
                last_id = int(ids_list[-1])
            else:
                last_id = 0  # Start from 0 if no IDs are present
            
            ids = [str(i) for i in range(last_id+1, last_id+1+len(self.token_split_text))]
            embeddings = self.embedding_function(self.token_split_text)
            collection.add(ids=ids, documents=self.token_split_text, embeddings= embeddings)
            self.collections = [collection.name for collection in self.client.list_collections()]
            self.token_split_text = None
            message_2 = "Document Processed Successfully"
        else:
            message_2 = "Please upload a file first!! and try again"
        return message_2


    def generate_initial_answer(self):
        augmentation_chain = self.prompt1 | self.llm
        response = augmentation_chain.invoke(
            {"collection": self.selected_collection,
            "query": self.query})
        self.generated_answer = response.content
        return self.generated_answer
    
    def query_collection(self):
        augmented_query = self.query + "\n" + self.generated_answer
        results_list_of_lists = self.selected_collection.query(query_texts=augmented_query, n_results = self.num_results)['documents']
        # flatten the list and keep unique results only (useful in case of multiple queries)
        unique_results = set()
        for res_list in results_list_of_lists:
            for res in res_list:
                unique_results.add(res)
        self.results = list(unique_results)
        # Get only top n answers
        self.results = self.results[:self.num_top_results]
        return self.results
    
    def get_final_answer(self):
        final_chain = self.prompt2 | self.llm
        info = "\n\n".join(self.results)
        response = final_chain.invoke(
            {"collection": self.selected_collection,
             "query": self.query,
             "information": info}
        )
        self.final_answer = response.content
        return self.final_answer
    
    def select_collection(self, collection):
        self.selected_collection = self.client.get_collection(name= collection)

    
    def chat_check(self, query):
        response = self.llm.invoke(query)
        return(response)


# bot = Rag_Bot("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")
# print(bot.chat_check("what is the capital of EGYPT"))