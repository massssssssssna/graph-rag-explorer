"""
api/index.py — Vercel Serverless Entrypoint for Flask app
"""
import sys
import os
from pathlib import Path

# Add root directory to sys.path and set working directory
root_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(root_dir))
os.chdir(str(root_dir))

from app import app

# Vercel WSGI entrypoint
app = app
