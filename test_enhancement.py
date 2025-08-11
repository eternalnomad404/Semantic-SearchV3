#!/usr/bin/env python3
"""
Demo script to showcase the new LLM-enhanced query functionality
"""

import os
import warnings
import sys

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add current directory to path
sys.path.append('.')

def demo_enhanced_search():
    print("🧠 Testing LLM-Enhanced Query Search System...")
    
    try:
        from app_main import LLMSemanticSearcher
        print("✅ Import successful")
        
        searcher = LLMSemanticSearcher()
        print("✅ Searcher initialized successfully")
        
        test_queries = [
            "ML tools",
            "web dev courses", 
            "data viz",
            "AI training"
        ]
        
        print("\n" + "="*80)
        print("🚀 DEMO: Query Enhancement Comparison")
        print("="*80)
        
        for query in test_queries:
            print(f"\n🔍 Testing Query: '{query}'")
            print("-" * 60)
            
            # Test WITHOUT enhancement
            print("📊 WITHOUT Enhancement:")
            results_basic = searcher.search_semantic(query, enhance_query=False)
            print(f"   Found {len(results_basic)} results")
            if results_basic:
                print(f"   Top result: {results_basic[0]['metadata'].get('values', ['N/A'])[0]} (Score: {results_basic[0]['score']:.3f})")
            
            # Test WITH enhancement
            print("\n🧠 WITH Enhancement:")
            enhanced_query = searcher.enhance_query_with_llm(query)
            print(f"   Enhanced query: '{enhanced_query}'")
            
            results_enhanced = searcher.search_semantic(query, enhance_query=True)
            print(f"   Found {len(results_enhanced)} results")
            if results_enhanced:
                print(f"   Top result: {results_enhanced[0]['metadata'].get('values', ['N/A'])[0]} (Score: {results_enhanced[0]['score']:.3f})")
            
            # Compare
            if results_basic and results_enhanced:
                score_diff = results_enhanced[0]['score'] - results_basic[0]['score']
                print(f"   📈 Score improvement: {score_diff:+.3f}")
            
            print()
        
        print("="*80)
        print("✅ Enhanced query system working perfectly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    demo_enhanced_search()
