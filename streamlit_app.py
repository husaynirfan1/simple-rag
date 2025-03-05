import pandas as pd
from datasets import load_dataset
import os
from fastcoref import LingMessCoref
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pymilvus.model.reranker import BGERerankFunction
from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    WeightedRanker,
    Collection,
    AnnSearchRequest,
    RRFRanker,
    connections,
)
from typing import List, Dict
from datetime import datetime, timezone
import re
from ollama import chat
from ollama import ChatResponse
from pymilvus import MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from telegram import Update
from functools import wraps
from telegram.constants import ChatAction
from telegram.ext import ContextTypes
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from functools import wraps
import asyncio
import telegram
from difflib import SequenceMatcher
import streamlit as st
import time 
from bs4 import BeautifulSoup

st.title("AlbAi V3")

chat_history = []
print(chat_history)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"], unsafe_allow_html=True)

# Connect to Zilliz Cloud (Milvus)
client = MilvusClient(
    uri="URI",
    token="TOKEN"
)
# Connect to Milvus server (Zilliz Cloud)
connections.connect(
    alias="default", 
    uri="URI", 
    token="TOKEN"
)

#query = input("Enter your query: ")
 
embedding_function = BGEM3EmbeddingFunction(use_fp16=False, device="cuda:0")
dense_dim = embedding_function.dim["dense"]

#set current date
current_date = datetime.now().strftime("%Y-%m-%d")
print(current_date)  # Outputs: ex :11:01 AM on Tuesday, November 28th 2023

def get_model():
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cuda:0")
    return ef

col = Collection("news_mideast")

def send_typing_action(func):
    """Sends typing action while processing the func command."""

    @wraps(func)
    async def command_func(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        # Send typing action
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING
        )
        # Call the original function
        return await func(update, context, *args, **kwargs)

    return command_func


from fastcoref import LingMessCoref

def resolve_chat_coref(chat_history, context_window=4):
    """
    Resolves coreferences in a multi-turn chat history using fastcoref with LingMess.
    
    Args:
        chat_history: List of dicts with 'role' (user/assistant) and 'content' keys, modified in place.
        context_window: Number of previous turns for context (default: 4).
        
    Returns:
        List of dicts with resolved content, ensuring proper coreference resolution.
    """
    # Initialize the LingMessCoref model
    model = LingMessCoref(device='cuda:0')  # Use GPU if available

    resolved_history = []
    i = 0

    while i < len(chat_history):
        # Start with the current turn
        current_turn = chat_history[i]
        current_role = current_turn['role']
        
        # Define context window including grouped turns
        start_idx = max(0, i - context_window)
        window_history = chat_history[start_idx:i+1]  # Include up to the current turn in context
        
        # Combine window messages into a single text
        full_text = " ".join([f"{msg['role']}: {msg['content']}" for msg in window_history])
        
        # Get coreference clusters using fastcoref
        preds = model.predict(texts=[full_text])
        clusters = preds[0].get_clusters(as_strings=False)  # Extract clusters from the model output
        
        # Build replacements
        replacements = {}
        for cluster in clusters:
            sorted_mentions = sorted(cluster, key=lambda x: x[0], reverse=True)
            main_mention = sorted_mentions[0]
            main_text = full_text[main_mention[0]:main_mention[1]]
            for mention in sorted_mentions[1:]:
                replacements[(mention[0], mention[1])] = main_text
        
        # Resolve the current turn's content
        resolved_content = current_turn['content']
        current_start = len(" ".join([f"{msg['role']}: {msg['content']}" for msg in window_history[:-1]])) + 1
        current_end = current_start + len(f"{current_role}: {current_turn['content']}")
        
        for (mention_start, mention_end), replacement in replacements.items():
            if current_start <= mention_start < current_end:  # Within the current turn
                rel_start = mention_start - current_start - len(f"{current_role}: ")
                rel_end = mention_end - current_start - len(f"{current_role}: ")
                if 0 <= rel_start < len(resolved_content):
                    if rel_end > len(resolved_content):
                        rel_end = len(resolved_content)
                    resolved_content = (
                        resolved_content[:rel_start] + replacement + resolved_content[rel_end:]
                    )
        
        # Merge and deduplicate lines within resolved_content
        lines = resolved_content.split("\n")
        unique_lines = []
        seen = set()
        for line in lines:
            clean_line = line.strip()
            if clean_line and clean_line not in seen:
                seen.add(clean_line)
                unique_lines.append(clean_line)
        merged_content = " ".join(unique_lines)  # Join with spaces for a single coherent turn
        
        # Update chat_history in place
        chat_history[i]['content'] = merged_content
        resolved_history.append({'role': current_role, 'content': merged_content})
        
        # Move to the next turn
        i += 1
    
    return resolved_history
    
def chat_content_to_string(resolved_chat):
    user_contents = [msg['content'] for msg in resolved_chat if msg['role'] == 'user']
    return user_contents[-1] if user_contents else ""
    
@send_typing_action  
async def handle_message(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    chat_id = update.message.chat_id
    chat_history.append({"role": "user", "content": user_message})
    # Resolve coreferences in-place
    resolve_chat_coref(chat_history)  # chat_history is now resolved
    stringofquery = chat_content_to_string(chat_history)
    
    # Process the resolved query
    response = await process_user_query(stringofquery, chat_history)
    escaped_response = ''.join(f'\\{char}' if char in {'#', '*', '_', '`', '[', ']', '(', ')', '>', '+', '-', '|', '{', '}', '.', '!'} else char for char in response)
    
    #await context.bot.send_message(chat_id=chat_id, text=escaped_response, parse_mode=telegram.constants.ParseMode.MARKDOWN_V2)
    #await context.bot.send_message(chat_id=chat_id, text=escaped_response, parse_mode=telegram.constants.ParseMode.HTML)
    await update.message.reply_text(response)

#------------------------------------------------

def split_html_string(html_string):
    """
    Splits an HTML string into a list of complete tag elements (including content).
    
    Args:
        html_string (str): The HTML string to split.
    
    Returns:
        list: A list of strings, each representing a complete HTML tag with its content.
    """
    # Parse the HTML string
    soup = BeautifulSoup(html_string, 'html.parser')
    
    # List to store complete tag strings
    tag_elements = []
    
    # Iterate over top-level elements
    for element in soup.contents:
        if hasattr(element, 'name') and element.name:  # Check if it's a tag
            # Convert the entire element (including content) back to a string
            tag_elements.append(str(element))
    
    return tag_elements

# Streamed response emulator
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def write_stream(stream):
    result = ""
    container = st.empty()
    for chunk in stream:
        result += chunk
        container.write(result, unsafe_allow_html=True)

def get_stream(response):
    ht = split_html_string(response)
    for chunk in ht:
        time.sleep(1)
        yield chunk


async def handle_message_st() -> None:
    if user_input := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(user_input)
                
        st.session_state.messages.append({"role": "user", "content": user_input})    
        chat_history.append({"role": "user", "content": user_input})
        # Resolve coreferences in-place
        resolvedc = resolve_chat_coref(chat_history)  # chat_history is now resolved
        print(resolvedc)
        stringofquery = chat_content_to_string(resolvedc)
        
        with st.spinner("Generating response..."):
            try:
            # Process the resolved query
                response = await process_user_query(stringofquery, resolvedc)
                
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.write_stream(response_generator(response))
                    #rchunks = get_stream(response)
                    #write_stream(stream=rchunks)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")


#-----------------------------------------------

def get_last_message(chat_history):
    # Return the last message in the chat history
    return chat_history[-1]["content"] if chat_history else ""

def hybrid_search(query_embeddings, sparse_weight=1.0, dense_weight=1.0):
    sparse_search_params = {"metric_type": "IP"}
    sparse_req = AnnSearchRequest(
        query_embeddings["sparse"], "sparse_vector", sparse_search_params, limit=10
    )
    dense_search_params = {"metric_type": "IP"}
    dense_req = AnnSearchRequest(
        query_embeddings["dense"], "dense_vector", dense_search_params, limit=10
    )
    rerank = RRFRanker(k=10)
    
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=10, output_fields=["text"]
    )

    # for result in results_rerank:
    #         print(f"Index: {result.index}")
    #         print(f"Score: {result.score:.6f}")
    #         print(f"Text: {result.text}\n")
    
    # combined_reranked_results = []

    # for comb in results_rerank:
    #     combined_reranked_results.append("\n"+comb.text)
    

    if len(res):
        return [hit.fields["text"] for hit in res[0]]
    else:
        return []

def build_user_prompt(retrieved_context, current_user_input, conversation_history="", current_date="February 23, 2025"):
    # System message
    system_message = """
    You are an AI assistant designed to provide accurate, helpful, and clear responses. The provided context is real-time news and articles from genuine sources. Use the provided context and user input to answer questions logically and thoroughly. Avoid speculation and unsupported claims.
    """
    
    # Combine conversation history with retrieved context
    full_context = f"""
    {conversation_history}
    {retrieved_context}
    """
    
    # User message
    user_message = f"""
    Context:
    {full_context}

    Question: {current_user_input}

    Instructions:
    1. Answer the question using the most recent relevant information up to {current_date}. If the context lacks sufficient data, state this clearly.
    2. Rephrase the question if needed to ensure clarity, but retain all original details.
    3. Format your response using HTML tags for readability.
    4. Include specific dates or titles from the context when relevant.
    5. Avoid making up information. If uncertain, say "I don't know."
    """
    
    # Build the prompt
    prompt = f"System: {system_message}\n\nUser: {user_message}"
    
    return prompt

async def process_user_query(query: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Processes a resolved user query using embeddings, hybrid search, reranking, and LLM response,
    balancing multi-turn context with focus on the current query.
    
    Args:
        query (str): Resolved query string from chat_content_to_string
        chat_history (List[Dict[str, str]]): Persistent chat history with resolved content
        
    Returns:
        str: Cleaned response from the LLM
    """
    # Embed the resolved query
    try:
        ef = get_model()  # Assume synchronous; wrap in asyncio.to_thread if async
        print(f"Query Embed (TOBE):\n{query}")
        query_embeddings = ef.encode_queries([query])
    except Exception as e:
        return f"Error in getting model or embeddings: {str(e)}"

    # Initialize the BGE re-ranker
    try:
        bge_rf = BGERerankFunction(
            model_name="BAAI/bge-reranker-v2-m3",
            device="cuda:0"
        )
        search_results = hybrid_search(query_embeddings=query_embeddings, sparse_weight=0.7, dense_weight=1.0)
    except Exception as e:
        return f"Error in hybrid search or reranking: {str(e)}"

    # Perform reranking
    try:
        results_rerank = bge_rf(
            query=query,
            documents=search_results,
            top_k=3,
        )
    except Exception as e:
        return f"Error in re-ranking results: {str(e)}"

    # Debug output
    for result in results_rerank:
        print(f"Index: {result.index}")
        print(f"Score: {result.score:.6f}")
        print(f"Text: {result.text}\n")

    # Combine reranked results into context
    combined_reranked_results = [f"\n{comb.text}" for comb in results_rerank]
    context = "\n".join(combined_reranked_results)

    # Build the user prompt with context and query
    USER_PROMPT = build_user_prompt(context, chat_history, query)

    # Prepare messages with a sliding window of recent turns
    # Use last 4 entries (2 user + 2 assistant turns) to balance context and relevance
    # context_window_size = 4
    # recent_history = chat_history[-context_window_size:] if len(chat_history) > context_window_size else chat_history
    # messages = recent_history + [{"role": "user", "content": USER_PROMPT}]
    
    # Call the chat API with limited history plus the new prompt
    try:
        response: ChatResponse = chat(
            model="qwen2.5:14b",
            messages=chat_history
        )
    except Exception as e:
        return f"Error in generating chat response: {str(e)}"
    
    response_content = response["message"]["content"]

    # Update chat_history with the current query and cleaned response
    # Only append the query if it’s not already the last user message in chat_history
    if not chat_history or chat_history[-1]['role'] != 'user' or chat_history[-1]['content'] != query:
        chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": response_content})
    
    # Debug print
    print(response_content)
    return response_content

# Start command handler
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Hello! I am AlbAi, your news companion. Ask me middle east news.')

# Start command handler
async def specs(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Qwen2.5:14B, Hybrid BGE M3 Embedding(Milvus), News Source: PressTV, Almanar')

def main():
    # Set up the Application with your bot token
    #application = Application.builder().token("7559735708:AAG9SqxYi0PsvJsuopEu67_ZLYoRt0PiSiY").build()
    
    # Add handlers for commands and messages
    #application.add_handler(CommandHandler("start", start))
    #application.add_handler(CommandHandler("specs", specs))
    #application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    asyncio.run(handle_message_st())
    # Start the bot
    #application.run_polling()

if __name__ == '__main__':
    main()
