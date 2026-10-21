# -*- coding: utf-8 -*-
"""Retriever access point for the mathematics knowledge base."""

from agentic_rag.math_retriever import math_retriever


def get_math_retriever():
    return math_retriever