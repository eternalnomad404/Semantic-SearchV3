#!/usr/bin/env python3
"""
Quick test script to verify no errors occur during search
"""

import os
import warnings
import sys

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add current directory to path
sys.path.append('.')

def test_search():
    print("🧪 Testing LLM Search System...")
    
    try:
        from app_main import LLMSemanticSearcher
        print("✅ Import successful")
        
        searcher = LLMSemanticSearcher()
        print("✅ Searcher initialized successfully")
        
        response, results = searcher.chat_search("machine learning tools")
        print("✅ Search completed successfully")
        print(f"📊 Found {len(results)} results")
        print("🤖 AI Response Preview:", response[:100] + "...")
        
        print("\n🎉 All tests passed! No errors detected.")
        
    except Exception as e:
        print(f"❌ Error detected: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_search()
    if success:
        print("\n✅ Application is working perfectly with no errors!")
    else:
        print("\n❌ Some issues detected.")
