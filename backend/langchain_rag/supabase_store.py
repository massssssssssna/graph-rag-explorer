"""
backend/langchain_rag/supabase_store.py
Supabase Cloud document metadata storage with graceful fallback.
"""
import logging
from typing import List, Dict, Any
import time

import config

logger = logging.getLogger(__name__)

_supabase_client = None
_local_doc_db: List[Dict[str, Any]] = []

def get_supabase_client():
    """Returns initialized Supabase Client if credentials are provided."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    if config.SUPABASE_URL and config.SUPABASE_KEY:
        try:
            from supabase import create_client
            _supabase_client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            logger.info("Connected to Supabase Cloud at %s", config.SUPABASE_URL)
            return _supabase_client
        except Exception as exc:
            logger.warning("Supabase connection failed: %s. Using in-memory document registry.", exc)
    return None

def save_document_metadata(doc_id: str, filename: str, file_type: str, num_chunks: int, size_bytes: int) -> Dict[str, Any]:
    """Store document metadata in Supabase or local registry."""
    record = {
        "id": doc_id,
        "filename": filename,
        "file_type": file_type,
        "num_chunks": num_chunks,
        "size_bytes": size_bytes,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    client = get_supabase_client()
    if client:
        try:
            client.table(config.SUPABASE_TABLE).insert(record).execute()
            logger.info("Saved metadata for '%s' in Supabase table '%s'", filename, config.SUPABASE_TABLE)
            return record
        except Exception as exc:
            logger.warning("Failed to insert record into Supabase (%s), saving locally.", exc)

    _local_doc_db.append(record)
    return record

def list_documents() -> List[Dict[str, Any]]:
    """Fetch list of all ingested documents."""
    client = get_supabase_client()
    if client:
        try:
            res = client.table(config.SUPABASE_TABLE).select("*").order("created_at", desc=True).execute()
            if res and res.data:
                return res.data
        except Exception as exc:
            logger.warning("Failed to fetch documents from Supabase: %s", exc)

    return _local_doc_db

def delete_document_metadata(doc_id: str) -> bool:
    """Delete document metadata record."""
    global _local_doc_db
    client = get_supabase_client()
    if client:
        try:
            client.table(config.SUPABASE_TABLE).delete().eq("id", doc_id).execute()
            logger.info("Deleted document %s from Supabase", doc_id)
        except Exception as exc:
            logger.warning("Supabase delete failed: %s", exc)

    _local_doc_db = [d for d in _local_doc_db if d.get("id") != doc_id]
    return True
