"""
api/index.py — Vercel Serverless Entrypoint for Flask app
"""
import sys
from pathlib import Path

# Add root directory to sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from app import app

# Vercel serverless handler
app = app
