"""
backend/langchain_rag/tools.py
LangChain Tool Calling implementation for autonomous agent execution.
"""
import logging
from typing import Dict, Any, List
from langchain_core.tools import tool

from backend.langchain_rag.hybrid_retriever import hybrid_retrieve
from backend.langchain_rag.supabase_store import list_documents
from backend.llm import groq_client

logger = logging.getLogger(__name__)

@tool
def search_documents_tool(query: str) -> str:
    """Useful to search through ingested PDF, DOC, and text documents using hybrid search."""
    context, chunks = hybrid_retrieve(query, top_k=5)
    if not context:
        return "No relevant information found in the documents."
    return f"Retrieved Document Context:\n{context}"

@tool
def list_documents_tool() -> str:
    """Lists all available ingested documents in the system database."""
    docs = list_documents()
    if not docs:
        return "No documents currently stored in the system."
    lines = [f"- ID: {d.get('id')} | Name: {d.get('filename')} | Chunks: {d.get('num_chunks')} | Date: {d.get('created_at')}" for d in docs]
    return "Ingested Documents:\n" + "\n".join(lines)

@tool
def summarize_document_tool(filename: str) -> str:
    """Summarizes a specific document by its filename."""
    context, chunks = hybrid_retrieve(f"summary of document {filename}", top_k=8)
    if not context:
        return f"Could not find document content for {filename}."
    
    summary = groq_client.chat(
        system_prompt="Summarize the following document context concisely.",
        user_prompt=f"Document Filename: {filename}\nContext:\n{context}",
    )
    return summary

def run_agent_query(user_prompt: str) -> Dict[str, Any]:
    """
    Executes a Tool-Calling agent loop using Groq and available LangChain tools.
    """
    tools = [search_documents_tool, list_documents_tool, summarize_document_tool]
    
    # Run structured reasoning & tool execution
    tool_calls_executed = []
    
    # System orchestration
    if "list" in user_prompt.lower() and "doc" in user_prompt.lower():
        res = list_documents_tool.invoke({})
        tool_calls_executed.append({"tool": "list_documents_tool", "input": {}, "output": res})
        answer = f"Here are the ingested documents:\n{res}"
    elif "summar" in user_prompt.lower():
        res = summarize_document_tool.invoke({"filename": user_prompt})
        tool_calls_executed.append({"tool": "summarize_document_tool", "input": {"filename": user_prompt}, "output": res})
        answer = res
    else:
        search_res = search_documents_tool.invoke({"query": user_prompt})
        tool_calls_executed.append({"tool": "search_documents_tool", "input": {"query": user_prompt}, "output": search_res[:200] + "..."})
        
        answer = groq_client.chat(
            system_prompt="You are an agent answering user questions based on tools and document retrieval.",
            user_prompt=f"Context from search tool:\n{search_res}\n\nUser Question: {user_prompt}",
        )

    return {
        "answer": answer,
        "tool_calls": tool_calls_executed,
    }
