#!/usr/bin/env python3
"""
Demo script to showcase the LLM-powered search functionality
"""

from app_main import LLMSemanticSearcher
import time

def demo_search():
    print("🤖 Initializing AI-Powered Search Assistant...")
    searcher = LLMSemanticSearcher()
    
    demo_queries = [
        "What are the best machine learning tools available?",
        "Find Python training courses for beginners",
        "Show me data visualization tools",
        "I need cybersecurity training programs"
    ]
    
    print("\n" + "="*80)
    print("🚀 DEMO: AI-Powered Semantic Search with LLM Responses")
    print("="*80)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 60)
        
        start_time = time.time()
        response, results = searcher.chat_search(query)
        search_time = time.time() - start_time
        
        print(f"🤖 AI Response (Generated in {search_time:.2f}s):")
        print(response)
        print(f"\n📊 Found {len(results)} relevant results")
        
        if i < len(demo_queries):
            print("\n" + "="*80)

if __name__ == "__main__":
    demo_search()
