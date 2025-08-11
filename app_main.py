# app_main.py

import os
import faiss
import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import time
import warnings
from typing import List, Dict, Any

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

class LLMSemanticSearcher:
    """
    An LLM-powered semantic searcher using FAISS, SentenceTransformer, and Groq API.
    Provides intelligent, conversational responses to search queries.
    """
    def __init__(self, 
                 index_path: str = "vectorstore/faiss_index.index", 
                 metadata_path: str = "vectorstore/metadata.json",
                 model_name: str = "all-MiniLM-L6-v2"):
        # Load FAISS index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.index = faiss.read_index(index_path)

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata JSON not found at {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Load embedding model with error suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = SentenceTransformer(model_name)
        
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.groq_client = Groq(api_key=api_key)

    def enhance_query_with_llm(self, original_query: str) -> str:
        """
        Use Groq LLM to enhance and expand the user query for better semantic search.
        This helps FAISS find more relevant results by understanding user intent.
        """
        enhancement_prompt = f"""You are a query enhancement assistant. Your job is to analyze a user's search query and expand it with relevant synonyms, related terms, and context to improve semantic search results.

Guidelines:
1. Keep the original intent intact
2. Add relevant synonyms and related terms
3. Include technical terms and common variations
4. Consider different ways people might describe the same thing
5. Keep it concise but comprehensive
6. Focus on terms that would appear in tools, services, or course descriptions

Original Query: "{original_query}"

Provide an enhanced version that includes the original query plus relevant synonyms and related terms. Return ONLY the enhanced query text, no explanations."""

        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Updated to supported model
                messages=[
                    {"role": "user", "content": enhancement_prompt}
                ],
                temperature=0.3,  # Lower temperature for more focused enhancement
                max_tokens=150,   # Shorter response for query enhancement
                top_p=0.8
            )
            
            enhanced_query = completion.choices[0].message.content.strip()
            return enhanced_query
            
        except Exception as e:
            # If enhancement fails, return original query
            print(f"Query enhancement failed: {e}")
            return original_query

    def search_semantic(self, query: str, k: int = 15, min_score: float = 0.25, enhance_query: bool = True) -> List[Dict[str, Any]]:
        """
        Perform semantic search and return relevant results.
        Now with optional LLM query enhancement for better results.
        """
        # Enhance query with LLM if enabled
        search_query = query
        if enhance_query:
            enhanced_query = self.enhance_query_with_llm(query)
            search_query = enhanced_query
            # Store enhancement info for display (but don't print in Streamlit context)
            
        # Encode query and search in FAISS
        query_vector = self.model.encode([search_query])
        distances, indices = self.index.search(query_vector, k * 2)

        results: List[Dict[str, Any]] = []
        seen_keys: set[tuple] = set()

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            item = self.metadata[idx]

            # Unique key to avoid duplicates
            key = tuple(str(v) for v in item.get('values', []))
            if key in seen_keys:
                continue

            score = 1 / (1 + dist)
            if score < min_score:
                continue

            results.append({
                'metadata': item,
                'score': float(score),
                'original_query': query,  # Store original query
                'enhanced_query': search_query  # Store enhanced query
            })
            seen_keys.add(key)

        # Sort by score descending and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]

    def generate_llm_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Generate an intelligent LLM response based on search results.
        """
        # Prepare context from search results
        context_items = []
        for i, result in enumerate(search_results[:10], 1):  # Use top 10 results
            metadata = result['metadata']
            values = metadata.get('values', [])
            headers = metadata.get('column_headers', [])
            source = metadata.get('sheet', 'Unknown')
            score = result['score']
            
            # Format the item information
            item_info = []
            for header, value in zip(headers, values):
                if value and str(value).strip():
                    item_info.append(f"{header}: {value}")
            
            context_items.append(f"{i}. [{source.upper()}] {' | '.join(item_info)} (Relevance: {score:.3f})")

        context_text = "\n".join(context_items)
        
        # Create the prompt for the LLM
        system_prompt = """You are an intelligent search assistant specializing in tools, service providers, and training courses. 
Your role is to provide helpful, comprehensive, and conversational responses based on the search results provided.

Guidelines:
1. Analyze the user's query and provide a direct, helpful answer
2. Use the search results to give specific recommendations
3. Organize information clearly with categories when relevant
4. Mention relevance scores when highlighting top recommendations
5. Be conversational and engaging, not just a list
6. If results span multiple categories, organize your response accordingly
7. Provide actionable insights and next steps when appropriate
8. If no highly relevant results exist, acknowledge this and suggest alternative search terms

Always structure your response to be informative and easy to read."""

        user_prompt = f"""User Query: "{query}"

Search Results:
{context_text}

Please provide a comprehensive, intelligent response that addresses the user's query using the search results above. 
Be specific, helpful, and conversational in your response."""

        try:
            # Make API call to Groq
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Updated to supported model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
                top_p=0.9
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error generating LLM response: {str(e)}")
            # Fallback to a basic response
            return self._generate_fallback_response(query, search_results)

    def _generate_fallback_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Generate a fallback response if LLM fails.
        """
        if not search_results:
            return f"I couldn't find any relevant results for '{query}'. Please try different search terms or be more specific."
        
        response = f"Based on your search for '{query}', I found {len(search_results)} relevant results:\n\n"
        
        for i, result in enumerate(search_results[:5], 1):
            metadata = result['metadata']
            values = metadata.get('values', [])
            headers = metadata.get('column_headers', [])
            source = metadata.get('sheet', 'Unknown')
            score = result['score']
            
            response += f"{i}. **{source.title()}**: "
            item_details = []
            for header, value in zip(headers, values):
                if value and str(value).strip():
                    item_details.append(f"{value}")
            response += " | ".join(item_details)
            response += f" (Relevance: {score:.3f})\n"
        
        return response

    def chat_search(self, query: str, enhance_query: bool = True) -> tuple[str, List[Dict[str, Any]]]:
        """
        Perform a complete chat-based search with LLM response.
        Now supports optional query enhancement for better semantic search results.
        """
        # Get semantic search results (with optional enhancement)
        search_results = self.search_semantic(query, enhance_query=enhance_query)
        
        # Generate LLM response
        llm_response = self.generate_llm_response(query, search_results)
        
        return llm_response, search_results


@st.cache_resource
def initialize_searcher() -> LLMSemanticSearcher:
    """Initialize and cache the LLMSemanticSearcher resource."""
    return LLMSemanticSearcher()


def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(
        page_title="🤖 AI-Powered Search Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .search-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .llm-response {
        background: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin-bottom: 1rem;
    }
    .search-results {
        background: #f9f9f9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0; text-align: center;">🤖 AI-Powered Search Assistant</h1>
        <p style="color: white; margin: 0; text-align: center; opacity: 0.9;">
            Intelligent search across tools, service providers, and training courses
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize searcher and handle missing data
    try:
        searcher = initialize_searcher()
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}")
        st.info("Please make sure the vectorstore files are generated by running `generate_embeddings.py`")
        return
    except Exception as e:
        st.error(f"⚠️ Error initializing the search system: {e}")
        return

    # Search interface
    with st.container():
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            query = st.text_input(
                "💬 Ask me anything about tools, services, or training:",
                placeholder="Example: What are the best machine learning tools available?",
                key="search_query"
            )
        
        with col2:
            enhance_query = st.checkbox("🧠 Smart Query", value=True, 
                                      help="Use AI to enhance your search query for better results")
        
        with col3:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat interface
    if query and (search_button or query != st.session_state.get('last_query', '')):
        if len(query.strip()) < 3:
            st.warning("⚠️ Please enter a longer search query (at least 3 characters).")
        else:
            # Store the query
            st.session_state.last_query = query
            
            with st.spinner("🤖 AI is thinking and searching..."):
                start_time = time.time()
                
                try:
                    # Perform the chat search with optional query enhancement
                    if enhance_query:
                        st.info("🧠 AI is enhancing your query for better results...")
                    
                    llm_response, search_results = searcher.chat_search(query, enhance_query=enhance_query)
                    
                    search_time = time.time() - start_time
                    
                    # Display LLM Response
                    st.markdown('<div class="llm-response">', unsafe_allow_html=True)
                    st.markdown("### 🤖 AI Assistant Response")
                    
                    # Show query enhancement if it was used
                    if enhance_query and search_results and 'enhanced_query' in search_results[0]:
                        original = search_results[0]['original_query']
                        enhanced = search_results[0]['enhanced_query']
                        if original != enhanced:
                            with st.expander("🧠 View Query Enhancement", expanded=False):
                                st.markdown(f"**Original:** {original}")
                                st.markdown(f"**Enhanced:** {enhanced}")
                                st.caption("AI expanded your query to find more relevant results")
                    
                    st.markdown(llm_response)
                    st.markdown(f"*Response generated in {search_time:.2f} seconds*")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show detailed search results in an expander
                    if search_results:
                        with st.expander(f"📊 View Detailed Search Results ({len(search_results)} items found)", expanded=False):
                            st.markdown('<div class="search-results">', unsafe_allow_html=True)
                            
                            for i, res in enumerate(search_results, start=1):
                                metadata = res['metadata']
                                values = metadata.get('values', [])
                                headers = metadata.get('column_headers', [])
                                source = metadata.get('sheet', 'Unknown')
                                score = res['score']
                                
                                # Create a nice display for each result
                                with st.container():
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.markdown(f"**#{i} - {source.title()}**")
                                        for header, value in zip(headers, values):
                                            if value and str(value).strip():
                                                st.text(f"• {header}: {value}")
                                    
                                    with col2:
                                        st.metric("Relevance", f"{score:.3f}")
                                    
                                    if i < len(search_results):
                                        st.divider()
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during search: {str(e)}")
                    st.info("Please try again or rephrase your query.")

    # Sidebar with information and examples
    with st.sidebar:
        st.markdown("### 🎯 How to Use")
        st.markdown("""
        This AI-powered search assistant can help you find:
        
        **🛠️ Tools & Software**
        - Development tools
        - Analytics platforms
        - Design software
        
        **🏢 Service Providers**
        - Companies and vendors
        - Professional services
        - Consultants
        
        **📚 Training Courses**
        - Skills development
        - Professional certifications
        - Educational programs
        """)
        
        st.markdown("### 💡 Example Queries")
        st.markdown("""
        Try these example searches:
        
        - "What are the best data visualization tools?"
        - "Find Python training courses for beginners"
        - "Show me cloud service providers"
        - "I need machine learning tools"
        - "What design software is available?"
        - "Find cybersecurity training programs"
        
        💡 **Tip**: Just copy and paste any example into the search box above!
        """)
        
        st.markdown("---")
        st.markdown("### ⚡ Powered By")
        st.markdown("""
        - **🧠 Groq AI** - Lightning-fast LLM responses
        - **🔍 FAISS** - Semantic similarity search
        - **🤖 Sentence Transformers** - Neural embeddings
        - **⚡ Streamlit** - Interactive interface
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Search Statistics")
        if 'search_results' in locals() and search_results:
            st.metric("Results Found", len(search_results))
            avg_score = sum(r['score'] for r in search_results) / len(search_results)
            st.metric("Average Relevance", f"{avg_score:.3f}")


if __name__ == "__main__":
    main()
    