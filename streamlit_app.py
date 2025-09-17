# app.py
import streamlit as st
import pandas as pd
import os
import glob
from datetime import datetime
import google.generativeai as genai
import json

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
        
        # Convert date column to datetime if it exists
        if 'date' in combined_df.columns:
            combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
        
        st.session_state.data_loaded = True
        return combined_df
    else:
        return pd.DataFrame()

# Load data if not already loaded
if not st.session_state.data_loaded:
    st.session_state.session_data = load_data()

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
        if not st.session_state.session_data.empty:
            st.write(f"Total sessions: {len(st.session_state.session_data['date'].dt.date.unique())}")
        else:
            st.write("Total sessions: 0 (no data loaded)")
        st.write(f"Total topics: {len(st.session_data)}")
        
        # Show categories
        if 'category' in st.session_state.session_data.columns:
            st.write("Categories:")
            for category in st.session_state.session_data['category'].unique():
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

# Function to search session data
def search_sessions(query, df):
    results = pd.DataFrame()
    
    # Convert query to lowercase for case-insensitive search
    query_lower = query.lower()
    
    # Search in topic column
    if 'topic' in df.columns:
        topic_matches = df[df['topic'].str.lower().str.contains(query_lower, na=False)]
        results = pd.concat([results, topic_matches])
    
    # Search in explanation column
    if 'explanation' in df.columns:
        explanation_matches = df[df['explanation'].str.lower().str.contains(query_lower, na=False)]
        results = pd.concat([results, explanation_matches])
    
    # Search in category column
    if 'category' in df.columns:
        category_matches = df[df['category'].str.lower().str.contains(query_lower, na=False)]
        results = pd.concat([results, category_matches])
    
    # Remove duplicates
    results = results.drop_duplicates()
    
    return results

# Function to handle different query types
def process_query(query, df):
    # Check for date queries
    if any(word in query.lower() for word in ['date', 'session', 'discussed on', 'discussed in']):
        # Try to extract date information
        try:
            # Simple date extraction (this could be enhanced with more sophisticated NLP)
            date_str = None
            for word in query.split():
                try:
                    date_str = word
                    parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
                    break
                except:
                    continue
            
            if date_str:
                date_matches = df[df['date'].dt.date == parsed_date.date()]
                if not date_matches.empty:
                    return date_matches
        except:
            pass
    
    # Default to general search
    return search_sessions(query, df)

# Function to format response with session data
def format_response(results, query):
    if results.empty:
        return f"Sorry, I couldn't find information about '{query}' in the session materials. If you'd like to explore more, [click here](https://chat.openai.com/?q={query.replace(' ', '+')})."
    
    response = f"I found {len(results)} result(s) related to '{query}':\n\n"
    
    for _, row in results.iterrows():
        response += f"### {row.get('topic', 'No topic')}\n"
        response += f"**Date:** {row.get('date', 'No date')}\n"
        response += f"**Category:** {row.get('category', 'No category')}\n\n"
        response += f"**Explanation:** {row.get('explanation', 'No explanation available')}\n\n"
        
        if 'reference material' in row and pd.notna(row['reference material']):
            response += f"**Reference Material:** [Link]({row['reference material']})\n"
        
        if 'session recording' in row and pd.notna(row['session recording']):
            response += f"**Session Recording:** [Link]({row['session recording']})\n"
        
        response += "---\n\n"
    
    return response

# Function to get Gemini response
def get_gemini_response(query, context):
    if not st.session_state.api_key:
        return "Please enter your Google Gemini API key in the sidebar to use this feature."
    
    try:
        genai.configure(api_key=st.session_state.api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        You are a helpful assistant. Use only the data below to answer the user's question.
        
        User Query: {query}
        
        Available Data: {context}
        
        Do not generate any information not present in the data above.
        If the data doesn't contain information to answer the question, politely say so.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

# Chat input
if prompt := st.chat_input("Ask about session topics..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process query
    if not st.session_state.session_data.empty:
        results = process_query(prompt, st.session_state.session_data)
        
        # Prepare context for Gemini
        context = ""
        if not results.empty:
            for _, row in results.iterrows():
                context += f"Topic: {row.get('topic', '')}\n"
                context += f"Date: {row.get('date', '')}\n"
                context += f"Category: {row.get('category', '')}\n"
                context += f"Explanation: {row.get('explanation', '')}\n"
                context += f"Reference Material: {row.get('reference material', '')}\n"
                context += f"Session Recording: {row.get('session recording', '')}\n\n"
        
        # Get response from Gemini
        with st.spinner("Thinking..."):
            response = get_gemini_response(prompt, context)
        
        # If no specific data found, use the formatted response instead
        if results.empty and "couldn't find" in response.lower():
            response = format_response(results, prompt)
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
    
    # Allow filtering by category
    if 'category' in st.session_state.session_data.columns:
        categories = st.session_state.session_data['category'].unique()
        selected_category = st.selectbox("Filter by category:", ["All"] + list(categories))
        
        if selected_category != "All":
            filtered_data = st.session_state.session_data[st.session_state.session_data['category'] == selected_category]
        else:
            filtered_data = st.session_state.session_data
    else:
        filtered_data = st.session_state.session_data
    
    # Display topics
    for _, row in filtered_data.iterrows():
        with st.expander(f"{row.get('date', 'No date').strftime('%Y-%m-%d') if hasattr(row.get('date'), 'strftime') else row.get('date', 'No date')}: {row.get('topic', 'No topic')}"):
            st.write(f"**Category:** {row.get('category', 'No category')}")
            st.write(f"**Explanation:** {row.get('explanation', 'No explanation available')}")
            
            col1, col2 = st.columns(2)
            if 'reference material' in row and pd.notna(row['reference material']):
                col1.markdown(f"**Reference Material:** [Link]({row['reference material']})")
            if 'session recording' in row and pd.notna(row['session recording']):
                col2.markdown(f"**Session Recording:** [Link]({row['session recording']})")
else:
    st.warning("No session data available. Please add Excel files to the /data directory.")