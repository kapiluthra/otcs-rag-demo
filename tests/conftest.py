"""
conftest.py — shared pytest configuration and fixtures

Run tests:
    cd otcs-rag-demo
    pip install pytest
    pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
