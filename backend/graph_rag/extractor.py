"""
backend/graph_rag/extractor.py
Extracts entities and relationships from text using LangChain's LLMGraphTransformer.
Supports two modes: predefined schema and LLM-defined schema.
"""
import logging
from typing import List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_experimental.graph_transformers import LLMGraphTransformer
import config

logger = logging.getLogger(__name__)

# Initialize Groq LLM for extraction
llm = ChatGroq(
    api_key=config.GROQ_API_KEY,
    model=config.GROQ_MODEL,
    temperature=0
)

# 1. LLM-defined Schema
llm_transformer_unconstrained = LLMGraphTransformer(llm=llm)

# 2. Predefined Schema
# We define allowed nodes and relationships based on common domains (like tech, space, etc.)
allowed_nodes = ["Person", "Organization", "Location", "Technology", "Concept", "Service"]
allowed_relationships = ["FOUNDED_BY", "LOCATED_IN", "DEVELOPS", "USES", "PARTNERS_WITH", "COMPETES_WITH", "RELATES_TO"]

llm_transformer_constrained = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=allowed_nodes,
    allowed_relationships=allowed_relationships
)

def extract_graph_documents(text: str, use_predefined_schema: bool = True) -> List:
    """
    Extracts GraphDocuments from raw text.
    """
    if not text or not text.strip():
        return []

    docs = [Document(page_content=text)]
    
    try:
        if use_predefined_schema:
            graph_docs = llm_transformer_constrained.convert_to_graph_documents(docs)
            logger.info("Extracted %d graph documents using PREDEFINED schema.", len(graph_docs))
        else:
            graph_docs = llm_transformer_unconstrained.convert_to_graph_documents(docs)
            logger.info("Extracted %d graph documents using LLM-DEFINED schema.", len(graph_docs))
            
        return graph_docs
    except Exception as exc:
        logger.warning("LLM graph extraction skipped or failed: %s", exc)
        return []

