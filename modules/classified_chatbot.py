import json
import os
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any
import time
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

# Set up OpenRouter with OpenAI client
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings_client = OpenAI(api_key=OPENAI_API_KEY)

# Connect to Qdrant (local or cloud)
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a text using OpenAI's text-embeddings-small-3 model."""
    start_time = time.time()
    response = embeddings_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding_time = time.time() - start_time
    print(f"Embedding time: {embedding_time} seconds")
    return response.data[0].embedding

def search_qdrant_simple(query: str, collection_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Perform simple search in Qdrant for a single query."""
    # Generate embedding for the query
    embedding = generate_embedding(query)

    start_time = time.time()
    
    print("======================================================================",collection_name)
    print("======================================================================",query)
    # Perform search
    search_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=embedding,
        limit=limit,
        with_payload=True,
        score_threshold=0.4
    )
    print("======================================================================",search_results)
    search_time = time.time() - start_time
    print(f"Search time: {search_time} seconds")

    start_time_1 = time.time()
    results = []
    for scored_point in search_results.points:
        results.append({
            "id": scored_point.id,
            "score": scored_point.score,
            "payload": scored_point.payload
        })
    format_time = time.time() - start_time_1
    print(f"Format time: {format_time} seconds")

    return results

def generate_response(query: str, context: List[Dict[str, Any]], model: str = "openai/gpt-4o-mini", message_history=None,chat_summary=None) -> str:
    """Generate a response using OpenAI based on retrieved context and message history."""
    # Prepare context text from search results
    start_time = time.time()
    context_text = "\n\n".join([
        f"Document {i+1}:\nText: {item['payload']['text']}\nKeywords: {', '.join(item['payload']['keywords'])}"
        for i, item in enumerate(context)
    ])
    context_time = time.time() - start_time
    print(f"Context time: {context_time} seconds")

    system_prompt = """
    You are an authoritative academic assistant for Notre Dame University (NDU) providing precise information based on the retrieved documents.

    IMPORTANT GUIDELINES:
    1. Provide ONLY ONE definitive answer based on the highest relevance matches in the context.
    2. If multiple potential answers exist, choose the one with the strongest evidence in the retrieved documents.
    3. Maintain conversational context by referring to the message history when appropriate.

    ### PLEASE NOTE
    If you do not find anything in the reterived documents, you should say
    
    "It seems we do not have that information at the moment". DO NOT give/say anything else rathar than this.

    Your goal is to provide the single most accurate answer as if you were an official university representative.
    """

    print("Used Model: ", model)
    
    # Build messages array including history if available
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add up to 10 most recent messages from history if available
    if message_history and isinstance(message_history, list):
        # Only take the 10 most recent messages (5 exchanges)
        recent_history = message_history[-10:] if len(message_history) > 10 else message_history
        # Add them to our messages list
        for msg in recent_history:
            if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add the current query with context
    user_prompt = f"Question: {query}\n\nContext from university documents:\n{context_text}"
    messages.append({"role": "user", "content": user_prompt + "Mentioned is the chat summary to understand what previous conversation is about :" + str(chat_summary)})
    
    start_time_1 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2
    )
    response_time = time.time() - start_time_1
    print(f"Response time: {response_time} seconds")

    return response.choices[0].message.content

def rag_pipeline_simple(query: str, collection_name: str = "admission_course_guide", model: str = "openai/gpt-4o-mini", message_history=None,chat_summary=None):
    """Complete RAG pipeline from user query to response."""
    print(f"Original query: {query}")
    #print(f"message_history: {message_history}")

    # Search Qdrant with a single query
    search_results = search_qdrant_simple(query, collection_name, limit=3)

    # Generate response
    response = generate_response(query, search_results, model, message_history,chat_summary)

    return {
        "original_query": query,
        "search_results": search_results,
        "response": response
    }
