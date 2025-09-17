# app.py
import streamlit as st
import pandas as pd
import os
import glob
from datetime import datetime
import google.generativeai as genai

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
        
        # Convert Date column to datetime if it exists
        if 'Date' in combined_df.columns:
            combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
        
        st.session_state.data_loaded = True
        return combined_df
    else:
        return pd.DataFrame()

# Load data if not already loaded
if not st.session_state.data_loaded:
    st.session_state.session_data = load_data()

# Function to search session data
def search_sessions(query, df):
    results = pd.DataFrame()
    query_lower = query.lower()
    
    # Search in Topic column
    if 'Topic' in df.columns:
        topic_matches = df[df['Topic'].str.lower().str.contains(query_lower, na=False)]
        results = pd.concat([results, topic_matches])
    
    # Search in Explanation column
    if 'Explanation' in df.columns:
        explanation_matches = df[df['Explanation'].str.lower().str.contains(query_lower, na=False)]
        results = pd.concat([results, explanation_matches])
    
    # Search in Category column
    if 'Category' in df.columns:
        category_matches = df[df['Category'].str.lower().str.contains(query_lower, na=False)]
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
            # Simple date extraction
            date_str = None
            for word in query.split():
                try:
                    date_str = word
                    parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
                    break
                except:
                    continue
            
            if date_str:
                date_matches = df[df['Date'].dt.date == parsed_date.date()]
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

# Function to get Gemini response - UPDATED WITH LATEST MODEL
def get_gemini_response(query, context):
    if not st.session_state.api_key:
        return "Please enter your Google Gemini API key in the sidebar to use this feature."
    
    try:
        # Configure the API with the provided key
        genai.configure(api_key=st.session_state.api_key)
        
        # Use the latest available model - gemini-1.5-pro is recommended
        # Alternative models: 'gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-2.5-flash'
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = f"""
        You are a helpful assistant. Use only the data below to answer the user's question.
        
        User Query: {query}
        
        Available Data: {context}
        
        Do not generate any information not present in the data above.
        If the data doesn't contain information to answer the question, politely say so.
        """
        
        response = model.generate_content(
            prompt,
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            }
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
        # Check if Date column exists
        if 'Date' in st.session_state.session_data.columns:
            st.write(f"Total sessions: {len(st.session_state.session_data['Date'].dt.date.unique())}")
            st.write(f"Total topics: {len(st.session_state.session_data)}")
        else:
            st.write("Total sessions: Column 'Date' not found")
            st.write(f"Total topics: {len(st.session_state.session_data)}")
        
        # Show categories
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
        results = process_query(prompt, st.session_state.session_data)
        
        # Prepare context for Gemini
        context = ""
        if not results.empty:
            for _, row in results.iterrows():
                context += f"Topic: {row.get('Topic', '')}\n"
                
                date_val = row.get('Date', '')
                if hasattr(date_val, 'strftime'):
                    date_val = date_val.strftime('%Y-%m-%d')
                context += f"Date: {date_val}\n"
                
                context += f"Category: {row.get('Category', '')}\n"
                context += f"Explanation: {row.get('Explanation', '')}\n"
                context += f"Reference Material: {row.get('Reference Material', '')}\n"
                context += f"Session Recording: {row.get('Session Recording', '')}\n\n"
        
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
    if 'Category' in st.session_state.session_data.columns:
        categories = st.session_state.session_data['Category'].unique()
        selected_category = st.selectbox("Filter by category:", ["All"] + list(categories))
        
        if selected_category != "All":
            filtered_data = st.session_state.session_data[st.session_state.session_data['Category'] == selected_category]
        else:
            filtered_data = st.session_state.session_data
    else:
        filtered_data = st.session_state.session_data
    
    # Display topics
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