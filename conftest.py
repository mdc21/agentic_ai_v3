"""
conftest.py — Project root pytest configuration.

Adds the project root to sys.path so that `from app.xxx import ...`
works in all test files regardless of where pytest is invoked from.
"""
import sys
import os

# Ensure the project root (this file's directory) is always on sys.path
sys.path.insert(0, os.path.dirname(__file__))
