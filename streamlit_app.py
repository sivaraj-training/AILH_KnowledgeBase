# app.py
import streamlit as st
import pandas as pd
import os
import glob
from datetime import datetime
import google.generativeai as genai
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re

# Page configuration
st.set_page_config(
    page_title="Session Knowledge Base",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
    }
    .topic-card {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        background-color: #f9f9f9;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stExpander {
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    .response-container {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f8ff;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "session_data" not in st.session_state:
    st.session_state.session_data = pd.DataFrame()
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "document_store" not in st.session_state:
    st.session_state.document_store = []

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Function to load data from Excel files
def load_data():
    data_path = "data"
    excel_files = glob.glob(os.path.join(data_path, "*.xlsx"))
    
    if not excel_files:
        st.error("No Excel files found in the /data directory.")
        return pd.DataFrame()
    
    all_data = []
    for file in excel_files:
        try:
            df = pd.read_excel(file, engine='openpyxl')
            all_data.append(df)
        except Exception as e:
            st.error(f"Error reading {file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Convert Date column to datetime if it exists
        if 'Date' in combined_df.columns:
            combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
        
        st.session_state.data_loaded = True
        return combined_df
    else:
        return pd.DataFrame()

# Function to create FAISS index
def create_faiss_index(df):
    documents = []
    for _, row in df.iterrows():
        doc_text = f"Topic: {row.get('Topic', '')}. "
        doc_text += f"Explanation: {row.get('Explanation', '')}. "
        doc_text += f"Category: {row.get('Category', '')}. "
        
        date_val = row.get('Date', '')
        if hasattr(date_val, 'strftime'):
            doc_text += f"Date: {date_val.strftime('%Y-%m-%d')}. "
        else:
            doc_text += f"Date: {date_val}. "
        
        documents.append({
            'text': doc_text,
            'topic': row.get('Topic', ''),
            'date': date_val,
            'category': row.get('Category', ''),
            'explanation': row.get('Explanation', ''),
            'reference_material': row.get('Reference Material', ''),
            'session_recording': row.get('Session Recording', ''),
            'original_row': row  # Store the original row for direct access
        })
    
    # Create embeddings
    texts = [doc['text'] for doc in documents]
    embeddings = embedding_model.encode(texts)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    return index, documents

# Function to search using RAG
def rag_search(query, top_k=3):
    if st.session_state.faiss_index is None or not st.session_state.document_store:
        return []
    
    # Embed the query
    query_embedding = embedding_model.encode([query])
    
    # Search in FAISS index
    distances, indices = st.session_state.faiss_index.search(query_embedding.astype('float32'), top_k)
    
    # Retrieve relevant documents
    results = []
    for idx in indices[0]:
        if idx < len(st.session_state.document_store):
            results.append(st.session_state.document_store[idx])
    
    return results

# Function to handle date range queries directly from the dataframe
def handle_date_range_query(query, df):
    # Check for date range patterns
    query_lower = query.lower()
    
    # Pattern for "July 2025" type queries
    if 'july 2025' in query_lower:
        if 'Date' in df.columns:
            july_2025_data = df[
                (df['Date'].dt.year == 2025) & 
                (df['Date'].dt.month == 7)
            ]
            return july_2025_data
        return pd.DataFrame()
    
    # Pattern for specific date queries
    elif any(month in query_lower for month in ['january', 'february', 'march', 'april', 'may', 'june', 
                                              'july', 'august', 'september', 'october', 'november', 'december']):
        # Extract year and month from query
        year_match = re.search(r'20\d{2}', query)
        year = int(year_match.group()) if year_match else datetime.now().year
        
        month_dict = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        month = None
        for month_name, month_num in month_dict.items():
            if month_name in query_lower:
                month = month_num
                break
        
        if month and 'Date' in df.columns:
            month_data = df[
                (df['Date'].dt.year == year) & 
                (df['Date'].dt.month == month)
            ]
            return month_data
        return pd.DataFrame()
    
    # Pattern for year queries
    elif re.search(r'20\d{2}', query):
        year_match = re.search(r'20\d{2}', query)
        if year_match:
            year = int(year_match.group())
            if 'Date' in df.columns:
                year_data = df[df['Date'].dt.year == year]
                return year_data
        return pd.DataFrame()
    
    return None

# Load data and create index if not already loaded
if not st.session_state.data_loaded:
    st.session_state.session_data = load_data()
    if not st.session_state.session_data.empty:
        st.session_state.faiss_index, st.session_state.document_store = create_faiss_index(st.session_state.session_data)

# Function to format response with session data
def format_response(results, query, is_date_range=False):
    if not results and not is_date_range:
        return f"Sorry, I couldn't find information about '{query}' in the session materials."
    
    if is_date_range:
        if results.empty:
            return f"No sessions found for '{query}'."
        
        response = f"## Sessions in {query}:\n\n"
        for _, row in results.iterrows():
            response += f"### {row.get('Topic', 'No topic')}\n"
            
            date_val = row.get('Date', 'No date')
            if hasattr(date_val, 'strftime'):
                date_val = date_val.strftime('%Y-%m-%d')
            response += f"**Date:** {date_val}\n"
            
            response += f"**Category:** {row.get('Category', 'No category')}\n\n"
            response += f"**Explanation:** {row.get('Explanation', 'No explanation available')}\n\n"
            
            if 'Reference Material' in row and pd.notna(row['Reference Material']):
                response += f"**Reference Material:** [Link]({row['Reference Material']})\n"
            
            if 'Session Recording' in row and pd.notna(row['Session Recording']):
                response += f"**Session Recording:** [Link]({row['Session Recording']})\n"
            
            response += "---\n\n"
        
        return response
    
    else:
        response = f"I found {len(results)} result(s) related to '{query}':\n\n"
        for result in results:
            response += f"### {result.get('topic', 'No topic')}\n"
            
            date_val = result.get('date', 'No date')
            if hasattr(date_val, 'strftime'):
                date_val = date_val.strftime('%Y-%m-%d')
            response += f"**Date:** {date_val}\n"
            
            response += f"**Category:** {result.get('category', 'No category')}\n\n"
            response += f"**Explanation:** {result.get('explanation', 'No explanation available')}\n\n"
            
            if result.get('reference_material') and pd.notna(result['reference_material']):
                response += f"**Reference Material:** [Link]({result['reference_material']})\n"
            
            if result.get('session_recording') and pd.notna(result['session_recording']):
                response += f"**Session Recording:** [Link]({result['session_recording']})\n"
            
            response += "---\n\n"
        
        return response

# Function to get Gemini response with RAG
def get_gemini_response(query, context):
    if not st.session_state.api_key:
        return "Please enter your Google Gemini API key in the sidebar to use this feature."
    
    try:
        genai.configure(api_key=st.session_state.api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        You are a helpful assistant. Use only the data below to answer the user's question.
        
        User Query: {query}
        
        Available Data: {context}
        
        Do not generate any information not present in the data above.
        If the data doesn't contain information to answer the question, politely say so.
        Keep your response concise and focused on the available data.
        """
        
        response = model.generate_content(
            prompt,
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            },
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.1
            )
        )
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

# Sidebar for API key and information
with st.sidebar:
    st.title("🔑 API Configuration")
    
    # API key input
    api_key = st.text_input("Enter your Google Gemini API Key:", type="password", 
                           help="Get your API key from https://aistudio.google.com/app/apikey")
    
    if st.button("Save API Key"):
        if api_key:
            st.session_state.api_key = api_key
            st.success("API key saved for this session.")
        else:
            st.error("Please enter a valid API key.")
    
    st.divider()
    
    # Data information
    st.title("📊 Data Overview")
    if not st.session_state.session_data.empty:
        if 'Date' in st.session_state.session_data.columns:
            st.write(f"Total sessions: {len(st.session_state.session_data['Date'].dt.date.unique())}")
            st.write(f"Total topics: {len(st.session_state.session_data)}")
        else:
            st.write("Total sessions: Column 'Date' not found")
            st.write(f"Total topics: {len(st.session_state.session_data)}")
        
        if 'Category' in st.session_state.session_data.columns:
            st.write("Categories:")
            for category in st.session_state.session_data['Category'].unique():
                st.write(f"- {category}")
    else:
        st.warning("No session data loaded.")
    
    st.divider()
    
    # Help section
    st.title("❓ Help")
    st.info("""
    Ask questions about:
    - Specific sessions (e.g., 'What was discussed on Sep 1, 2025?')
    - Topics or keywords (e.g., 'Tell me about machine learning')
    - Categories (e.g., 'Show me all data science topics')
    - Date ranges (e.g., 'What was discussed in September 2025?')
    """)

# Main app area
st.markdown('<h1 class="main-header">🤖 Session Knowledge Base</h1>', unsafe_allow_html=True)
st.caption("Ask questions about past session topics and materials")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about session topics..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process query
    if not st.session_state.session_data.empty:
        # First check if this is a date range query
        date_range_results = handle_date_range_query(prompt, st.session_state.session_data)
        
        if date_range_results is not None:
            # This is a date range query, handle it directly
            response = format_response(date_range_results, prompt, is_date_range=True)
        else:
            # Use RAG for other types of queries
            relevant_docs = rag_search(prompt, top_k=5)
            
            # Prepare context for Gemini
            context = ""
            if relevant_docs:
                for doc in relevant_docs:
                    context += f"Topic: {doc.get('topic', '')}\n"
                    
                    date_val = doc.get('date', '')
                    if hasattr(date_val, 'strftime'):
                        date_val = date_val.strftime('%Y-%m-%d')
                    context += f"Date: {date_val}\n"
                    
                    context += f"Category: {doc.get('category', '')}\n"
                    context += f"Explanation: {doc.get('explanation', '')}\n"
                    context += f"Reference Material: {doc.get('reference_material', '')}\n"
                    context += f"Session Recording: {doc.get('session_recording', '')}\n\n"
            
            # Get response from Gemini
            with st.spinner("Thinking..."):
                if relevant_docs:
                    response = get_gemini_response(prompt, context)
                else:
                    response = format_response(relevant_docs, prompt)
    else:
        response = "No session data available. Please ensure Excel files are in the /data directory."
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display session data if available
if not st.session_state.session_data.empty:
    st.divider()
    st.subheader("📚 All Session Topics")
    
    if 'Category' in st.session_state.session_data.columns:
        categories = st.session_state.session_data['Category'].unique()
        selected_category = st.selectbox("Filter by category:", ["All"] + list(categories))
        
        if selected_category != "All":
            filtered_data = st.session_state.session_data[st.session_state.session_data['Category'] == selected_category]
        else:
            filtered_data = st.session_state.session_data
    else:
        filtered_data = st.session_state.session_data
    
    for _, row in filtered_data.iterrows():
        date_str = row.get('Date', 'No date')
        if hasattr(date_str, 'strftime'):
            date_str = date_str.strftime('%Y-%m-%d')
            
        with st.expander(f"{date_str}: {row.get('Topic', 'No topic')}"):
            st.write(f"**Category:** {row.get('Category', 'No category')}")
            st.write(f"**Explanation:** {row.get('Explanation', 'No explanation available')}")
            
            col1, col2 = st.columns(2)
            if 'Reference Material' in row and pd.notna(row['Reference Material']):
                col1.markdown(f"**Reference Material:** [Link]({row['Reference Material']})")
            if 'Session Recording' in row and pd.notna(row['Session Recording']):
                col2.markdown(f"**Session Recording:** [Link]({row['Session Recording']})")
else:
    st.warning("No session data available. Please add Excel files to the /data directory.")