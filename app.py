import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import json
import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor
import re
import base64
from io import StringIO

# Set up the Streamlit app
st.set_page_config(page_title="LLM Website Mention Tracker", layout="wide")
st.title("LLM Website Mention Tracker")

# Initialize data structure
def init_data():
    if 'tracking_data' not in st.session_state:
        # Load data if it exists
        if os.path.exists('llm_mention_data.json'):
            with open('llm_mention_data.json', 'r') as f:
                st.session_state.tracking_data = json.load(f)
        else:
            st.session_state.tracking_data = []
    
    if 'websites' not in st.session_state:
        st.session_state.websites = []
        
    if 'keywords' not in st.session_state:
        st.session_state.keywords = []
        
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            "openai": "",
            "anthropic": "",
            "gemini": "",
            "mistral": ""
        }
        
    if 'platforms_to_check' not in st.session_state:
        st.session_state.platforms_to_check = ["ChatGPT", "Claude", "Gemini", "Mistral"]
    
    # Add demo mode flag
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = True  # Set to True by default

# Call initialization
init_data()

# Function to query OpenAI API
def query_openai(keyword, api_key):
    if not api_key:
        return {"platform": "ChatGPT", "response": "No API key provided", "error": True}
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "gpt-4o-mini",  # Matches your curl command
        "store": True,           # Added this from your curl command
        "messages": [{"role": "user", "content": keyword}],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return {"platform": "ChatGPT", "response": content, "error": False}
        else:
            return {"platform": "ChatGPT", "response": f"Error: {response.status_code} - {response.text}", "error": True}
    except Exception as e:
        return {"platform": "ChatGPT", "response": f"Error: {str(e)}", "error": True}

# Function to query Claude API
def query_claude(keyword, api_key):
    if not api_key:
        return {"platform": "Claude", "response": "No API key provided", "error": True}
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"  # This might need updating but the older version often still works
    }
    
    data = {
        "model": "claude-3-7-sonnet-20250219",  # Updated to Claude 3.7 Sonnet
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": keyword}
        ]
    }
    
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["content"][0]["text"]
            return {"platform": "Claude", "response": content, "error": False}
        else:
            return {"platform": "Claude", "response": f"Error: {response.status_code} - {response.text}", "error": True}
    except Exception as e:
        return {"platform": "Claude", "response": f"Error: {str(e)}", "error": True}

# Function to query Gemini API
def query_gemini(keyword, api_key):
    if not api_key:
        return {"platform": "Gemini", "response": "No API key provided", "error": True}
    
    # Updated URL to use v1 instead of v1beta
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent?key={api_key}"
    
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": keyword
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1000
        }
    }
    
    try:
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                return {"platform": "Gemini", "response": content, "error": False}
            else:
                return {"platform": "Gemini", "response": "No content in response", "error": True}
        else:
            return {"platform": "Gemini", "response": f"Error: {response.status_code} - {response.text}", "error": True}
    except Exception as e:
        return {"platform": "Gemini", "response": f"Error: {str(e)}", "error": True}

# Function to query Mistral API
def query_mistral(keyword, api_key):
    if not api_key:
        return {"platform": "Mistral", "response": "No API key provided", "error": True}
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": keyword}],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return {"platform": "Mistral", "response": content, "error": False}
        else:
            return {"platform": "Mistral", "response": f"Error: {response.status_code} - {response.text}", "error": True}
    except Exception as e:
        return {"platform": "Mistral", "response": f"Error: {str(e)}", "error": True}

# Function to check if website is mentioned in response
def check_website_mention(response, website):
    # Remove http/https and www. for more flexible matching
    clean_website = re.sub(r'^(https?://)?(www\.)?', '', website.lower())
    
    # Different mention patterns
    exact_match = clean_website.lower() in response.lower()
    partial_match = re.search(r'(\S*%s\S*)' % re.escape(clean_website), response.lower()) is not None
    
    if exact_match:
        return 10  # Strong mention
    elif partial_match:
        return 7   # Partial mention
    else:
        return 0   # No mention

# Function to run queries across platforms
def run_llm_queries(keyword, websites, platforms_to_check):
    # Check if in demo mode
    if st.session_state.demo_mode:
        results = []
        # Use pre-recorded demo data
        if os.path.exists('demo_data.json'):
            with open('demo_data.json', 'r') as f:
                demo_data = json.load(f)
                
            # Filter demo data based on platforms
            demo_results = []
            for entry in demo_data.get('tracking_data', []):
                if entry.get('platform') in platforms_to_check:
                    # Clone the entry to avoid modifying the original
                    entry_copy = entry.copy()
                    # Override with the current keyword if needed
                    entry_copy['keyword'] = keyword
                    demo_results.append(entry_copy)
            
            # Check for each website
            for website in websites:
                for platform in platforms_to_check:
                    # Try to find a matching entry
                    found = False
                    for entry in demo_results:
                        if entry['platform'] == platform:
                            # Create a new result with the current website
                            new_result = entry.copy()
                            new_result['website'] = website
                            
                            # Calculate a score for this website
                            if website == 'example.com':
                                new_result['score'] = entry['score']  # Keep original score for example.com
                            else:
                                # For other websites, generate a random score with higher chance of 0
                                import random
                                new_result['score'] = random.choice([0, 0, 0, 0, 7, 10])
                            
                            results.append(new_result)
                            found = True
                            break
                    
                    # If no matching entry, create a synthetic one
                    if not found:
                        import random
                        score = random.choice([0, 0, 0, 7, 10]) if website == "example.com" else random.choice([0, 0, 0, 0, 7])
                        
                        results.append({
                            "platform": platform,
                            "keyword": keyword,
                            "website": website,
                            "score": score,
                            "response": f"This is a demo response for {keyword}. {' It mentions ' + website + ' as a valuable resource.' if score > 0 else ''}",
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "error": False
                        })
            
            return results
    
    # Original code for non-demo mode
    results = []
    
    platform_functions = {
        "ChatGPT": query_openai,
        "Claude": query_claude,
        "Gemini": query_gemini,
        "Mistral": query_mistral
    }
    
    platform_api_keys = {
        "ChatGPT": st.session_state.api_keys["openai"],
        "Claude": st.session_state.api_keys["anthropic"],
        "Gemini": st.session_state.api_keys["gemini"],
        "Mistral": st.session_state.api_keys["mistral"]
    }
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(platform_functions[platform], keyword, platform_api_keys[platform]): platform
            for platform in platforms_to_check if platform in platform_functions
        }
        
        for future in futures:
            platform = futures[future]
            try:
                response_data = future.result()
                
                if not response_data["error"]:
                    # Check each website's mention
                    for website in websites:
                        mention_score = check_website_mention(response_data["response"], website)
                        
                        results.append({
                            "platform": platform,
                            "keyword": keyword,
                            "website": website,
                            "score": mention_score,
                            "response": response_data["response"],
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                else:
                    # Add error result
                    for website in websites:
                        results.append({
                            "platform": platform,
                            "keyword": keyword,
                            "website": website,
                            "score": 0,
                            "response": response_data["response"],
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "error": True
                        })
            except Exception as e:
                # Add exception result
                for website in websites:
                    results.append({
                        "platform": platform,
                        "keyword": keyword,
                        "website": website,
                        "score": 0,
                        "response": f"Error: {str(e)}",
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "error": True
                    })
    
    return results

# Sidebar for configuration and tracking
with st.sidebar:
    st.header("Configuration")
    
# Add demo mode toggle
    demo_mode = st.checkbox("Demo Mode (No API Keys Required)", 
                           value=st.session_state.demo_mode,
                           help="Run with pre-recorded data instead of making real API calls")
    
    if demo_mode != st.session_state.demo_mode:
        st.session_state.demo_mode = demo_mode
        st.success(f"Demo mode {'enabled' if demo_mode else 'disabled'}")
    
    # If demo mode is enabled, show a notice
    if st.session_state.demo_mode:
        st.info("⚠️ Demo mode is enabled. No real API calls will be made. Sample data will be used instead.")

    # API Key Management
    with st.expander("API Keys (Required)"):
        openai_key = st.text_input("OpenAI API Key:", value=st.session_state.api_keys["openai"], type="password")
        anthropic_key = st.text_input("Anthropic API Key:", value=st.session_state.api_keys["anthropic"], type="password")
        gemini_key = st.text_input("Google Gemini API Key:", value=st.session_state.api_keys["gemini"], type="password")
        mistral_key = st.text_input("Mistral API Key:", value=st.session_state.api_keys["mistral"], type="password")
        
        if st.button("Save API Keys"):
            st.session_state.api_keys = {
                "openai": openai_key,
                "anthropic": anthropic_key,
                "gemini": gemini_key,
                "mistral": mistral_key
            }
            st.success("API keys saved!")
    
    # Website Management
    with st.expander("Manage Websites"):
        website_input = st.text_input("Website URL (e.g., example.com):")
        if st.button("Add Website"):
            if website_input and website_input not in st.session_state.websites:
                st.session_state.websites.append(website_input)
                st.success(f"Added {website_input}")
            else:
                st.error("Website required or already exists")

        if st.session_state.websites:
            websites_to_remove = st.multiselect(
                "Select websites to remove:",
                options=st.session_state.websites
            )
            if st.button("Remove Selected Websites"):
                for website in websites_to_remove:
                    st.session_state.websites.remove(website)
                st.success("Websites removed!")
                
        st.write("Current websites:")
        for website in st.session_state.websites:
            st.write(f"- {website}")
    
    # Keyword Management
    with st.expander("Manage Keywords"):
        keyword_input = st.text_input("Keyword or query:")
        if st.button("Add Keyword"):
            if keyword_input and keyword_input not in st.session_state.keywords:
                st.session_state.keywords.append(keyword_input)
                st.success(f"Added {keyword_input}")
            else:
                st.error("Keyword required or already exists")

        if st.session_state.keywords:
            keywords_to_remove = st.multiselect(
                "Select keywords to remove:",
                options=st.session_state.keywords
            )
            if st.button("Remove Selected Keywords"):
                for keyword in keywords_to_remove:
                    st.session_state.keywords.remove(keyword)
                st.success("Keywords removed!")
                
        # Bulk upload
        st.write("Or upload keywords in bulk:")
        uploaded_file = st.file_uploader("Upload CSV or TXT file with keywords", type=["csv", "txt"])
        if uploaded_file is not None:
            try:
                # Handle CSV
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    if len(df.columns) > 0:
                        keywords = df.iloc[:,0].tolist()
                    else:
                        keywords = []
                # Handle TXT (one keyword per line)
                else:
                    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    keywords = [line.strip() for line in stringio.readlines()]
                
                # Add keywords
                for keyword in keywords:
                    if keyword and keyword not in st.session_state.keywords:
                        st.session_state.keywords.append(keyword)
                
                st.success(f"Added {len(keywords)} keywords!")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        st.write("Current keywords:")
        if len(st.session_state.keywords) <= 10:
            for keyword in st.session_state.keywords:
                st.write(f"- {keyword}")
        else:
            st.write(f"- {len(st.session_state.keywords)} keywords configured")
    
    # Platform selection
    with st.expander("Select LLM Platforms"):
        platforms = ["ChatGPT", "Claude", "Gemini", "Mistral"]
        selected_platforms = st.multiselect(
            "Platforms to check:",
            options=platforms,
            default=st.session_state.platforms_to_check
        )
        
        if st.button("Update Platforms"):
            st.session_state.platforms_to_check = selected_platforms
            st.success("Platforms updated!")
    
    # Run Tracking
    st.header("Run Tracking")
    
    # Validate configuration before running
can_run = (
    len(st.session_state.websites) > 0 and
    len(st.session_state.keywords) > 0 and
    len(st.session_state.platforms_to_check) > 0
)

# Only check for API keys if not in demo mode
has_api_keys = True  # Default to True for demo mode
if not st.session_state.demo_mode:
    has_api_keys = False
    for platform in st.session_state.platforms_to_check:
        if platform == "ChatGPT" and st.session_state.api_keys["openai"]:
            has_api_keys = True
        elif platform == "Claude" and st.session_state.api_keys["anthropic"]:
            has_api_keys = True
        elif platform == "Gemini" and st.session_state.api_keys["gemini"]:
            has_api_keys = True
        elif platform == "Mistral" and st.session_state.api_keys["mistral"]:
            has_api_keys = True

can_run = can_run and has_api_keys

if not can_run:
    if st.session_state.demo_mode:
        st.warning("Please configure websites, keywords, and platforms before running tracking.")
    else:
        st.warning("Please configure websites, keywords, platforms, and at least one API key before running tracking.")

# Update the API usage warning
if st.session_state.demo_mode:
    st.info("Demo Mode: No API credits will be used - this is using pre-recorded data")
else:
    st.warning("Warning: Running queries will use API credits/tokens from your accounts!")
    
    # Select specific keywords and websites for this run
    selected_keywords = st.multiselect(
        "Select keywords for this run:",
        options=st.session_state.keywords,
        default=st.session_state.keywords[:min(5, len(st.session_state.keywords))]
    )
    
    selected_websites = st.multiselect(
        "Select websites for this run:",
        options=st.session_state.websites,
        default=st.session_state.websites
    )
    
    st.warning("Warning: Running queries will use API credits/tokens from your accounts!")
    
    if st.button("Run Tracking Now", disabled=not can_run or not selected_keywords or not selected_websites):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        total_queries = len(selected_keywords)
        
        for i, keyword in enumerate(selected_keywords):
            status_text.text(f"Running query {i+1}/{total_queries}: '{keyword}'")
            results = run_llm_queries(keyword, selected_websites, st.session_state.platforms_to_check)
            all_results.extend(results)
            progress_bar.progress((i + 1) / total_queries)
            time.sleep(1)  # Small delay to avoid rate limits
        
        # Add to tracking data
        st.session_state.tracking_data.extend(all_results)
        
        # Save to file
        with open('llm_mention_data.json', 'w') as f:
            json.dump(st.session_state.tracking_data, f)
        
        status_text.text("Tracking completed!")
        st.success(f"Successfully processed {len(selected_keywords)} keywords across {len(st.session_state.platforms_to_check)} platforms!")

    # Schedule option (information only in this version)
    st.info("Note: For automated scheduling, set up this script with a cron job or scheduler on your server.")
    
# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Detailed Results", "Responses", "Export"])

with tab1:
    st.header("Website Mention Dashboard")
    
    if st.session_state.tracking_data:
        # Convert to DataFrame for visualization
        df = pd.DataFrame(st.session_state.tracking_data)
        
        # Add date column extracted from timestamp
        if 'timestamp' in df.columns:
            df['date'] = df['timestamp'].str.split(' ').str[0]
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            website_filter = st.selectbox(
                "Select Website:",
                options=["All"] + list(df['website'].unique())
            )
        
        with col2:
            platform_filter = st.selectbox(
                "Select Platform:",
                options=["All"] + list(df['platform'].unique())
            )
        
        with col3:
            if 'date' in df.columns:
                date_filter = st.selectbox(
                    "Select Date:",
                    options=["All"] + sorted(list(df['date'].unique()), reverse=True)
                )
            else:
                date_filter = "All"
        
        # Apply filters
        filtered_df = df.copy()
        if website_filter != "All":
            filtered_df = filtered_df[filtered_df['website'] == website_filter]
        if platform_filter != "All":
            filtered_df = filtered_df[filtered_df['platform'] == platform_filter]
        if date_filter != "All" and 'date' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['date'] == date_filter]
        
        if not filtered_df.empty:
            # Keyword performance chart
            st.subheader("Keyword Performance")
            
            # Group by keyword and calculate average score
            keyword_performance = filtered_df.groupby('keyword')['score'].mean().sort_values(ascending=False).reset_index()
            keyword_performance['score'] = keyword_performance['score'].round(2)
            
            fig = px.bar(
                keyword_performance, 
                x='keyword', 
                y='score',
                title="Average Mention Score by Keyword",
                labels={'score': 'Average Mention Score', 'keyword': 'Keyword'},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Platform comparison
            if platform_filter == "All":
                st.subheader("Platform Comparison")
                
                platform_performance = filtered_df.groupby('platform')['score'].mean().sort_values(ascending=False).reset_index()
                platform_performance['score'] = platform_performance['score'].round(2)
                
                fig2 = px.bar(
                    platform_performance, 
                    x='platform', 
                    y='score',
                    title="Average Mention Score by Platform",
                    labels={'score': 'Average Mention Score', 'platform': 'Platform'},
                    height=400,
                    color='platform'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Time trend if we have timestamp data
            if 'date' in filtered_df.columns and date_filter == "All":
                st.subheader("Mention Trend Over Time")
                
                time_trend = filtered_df.groupby('date')['score'].mean().reset_index()
                time_trend = time_trend.sort_values('date')
                
                fig3 = px.line(
                    time_trend,
                    x='date',
                    y='score',
                    title="Average Mention Score Over Time",
                    labels={'score': 'Average Mention Score', 'date': 'Date'},
                    height=400,
                    markers=True
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            # Summary metrics
            st.subheader("Summary Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Mention Score", f"{filtered_df['score'].mean():.2f}")
            
            with col2:
                mention_rate = (filtered_df['score'] > 0).mean() * 100
                st.metric("Mention Rate", f"{mention_rate:.1f}%")
            
            with col3:
                strong_mention = (filtered_df['score'] >= 7).mean() * 100
                st.metric("Strong Mention Rate", f"{strong_mention:.1f}%")
            
            with col4:
                queries_count = len(filtered_df['keyword'].unique())
                st.metric("Keywords Tracked", queries_count)
        else:
            st.info("No data matches your filter criteria.")
    else:
        st.info("No tracking data yet. Configure your websites and keywords, then run tracking.")

with tab2:
    st.header("Detailed Results")
    
    if st.session_state.tracking_data:
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.tracking_data)
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            keyword_filter = st.multiselect(
                "Select Keywords:",
                options=list(df['keyword'].unique()),
                default=[]
            )
        
        with col2:
            website_filter = st.multiselect(
                "Select Websites:",
                options=list(df['website'].unique()),
                default=[]
            )
        
        with col3:
            platform_filter = st.multiselect(
                "Select Platforms:",
                options=list(df['platform'].unique()),
                default=[]
            )
        
        # Apply filters
        filtered_df = df.copy()
        if keyword_filter:
            filtered_df = filtered_df[filtered_df['keyword'].isin(keyword_filter)]
        if website_filter:
            filtered_df = filtered_df[filtered_df['website'].isin(website_filter)]
        if platform_filter:
            filtered_df = filtered_df[filtered_df['platform'].isin(platform_filter)]
        
        # Score range filter
        score_range = st.slider(
            "Filter by Score Range:",
            min_value=0,
            max_value=10,
            value=(0, 10)
        )
        filtered_df = filtered_df[(filtered_df['score'] >= score_range[0]) & (filtered_df['score'] <= score_range[1])]
        
        # Date filter if available
        if 'timestamp' in filtered_df.columns:
            filtered_df['date'] = filtered_df['timestamp'].str.split(' ').str[0]
            
            date_filter = st.multiselect(
                "Select Dates:",
                options=sorted(list(filtered_df['date'].unique()), reverse=True),
                default=[]
            )
            
            if date_filter:
                filtered_df = filtered_df[filtered_df['date'].isin(date_filter)]
        
        # Display results table
        if not filtered_df.empty:
            # Prepare display dataframe (excluding full response text)
            display_df = filtered_df[['keyword', 'website', 'platform', 'score', 'timestamp']]
            if 'error' in filtered_df.columns:
                display_df['has_error'] = filtered_df['error'].fillna(False)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Cross-tabulation view
            st.subheader("Cross-tabulation View")
            pivot_option = st.selectbox(
                "Select Pivot View:",
                options=["Platform vs Keyword", "Website vs Keyword", "Platform vs Website"]
            )
            
            if pivot_option == "Platform vs Keyword":
                pivot_df = pd.pivot_table(
                    filtered_df,
                    values='score',
                    index='platform',
                    columns='keyword',
                    aggfunc='mean',
                    fill_value=0
                ).round(1)
                st.dataframe(pivot_df, use_container_width=True)
            
            elif pivot_option == "Website vs Keyword":
                pivot_df = pd.pivot_table(
                    filtered_df,
                    values='score',
                    index='website',
                    columns='keyword',
                    aggfunc='mean',
                    fill_value=0
                ).round(1)
                st.dataframe(pivot_df, use_container_width=True)
            
            else:  # Platform vs Website
                pivot_df = pd.pivot_table(
                    filtered_df,
                    values='score',
                    index='platform',
                    columns='website',
                    aggfunc='mean',
                    fill_value=0
                ).round(1)
                st.dataframe(pivot_df, use_container_width=True)
        else:
            st.info("No data matches your filter criteria.")
    else:
        st.info("No tracking data yet. Configure your websites and keywords, then run tracking.")

with tab3:
    st.header("LLM Responses")
    
    if st.session_state.tracking_data:
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.tracking_data)
        
        # Filter controls
        col1, col2 = st.columns(2)
        
        with col1:
            keyword_select = st.selectbox(
                "Select Keyword:",
                options=list(df['keyword'].unique())
            )
        
        with col2:
            platform_select = st.selectbox(
                "Select Platform:",
                options=list(df['platform'].unique())
            )
        
        # Filter data
        response_df = df[(df['keyword'] == keyword_select) & (df['platform'] == platform_select)]
        
        if not response_df.empty:
            # Get the latest response
            latest_response = response_df.sort_values('timestamp', ascending=False).iloc[0]
            
            # Display response info
            st.subheader("Response Details")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Website", latest_response['website'])
            with col2:
                st.metric("Mention Score", latest_response['score'])
            with col3:
                st.metric("Timestamp", latest_response['timestamp'])
            
            # Show the full response
            st.subheader("Full LLM Response")
            
            # Check if the response is too long and truncate if needed
            response_text = latest_response['response']
            if len(response_text) > 30000:  # Streamlit text area has limits
                response_text = response_text[:30000] + "... (response truncated due to length)"
            
            st.text_area("Response Text", response_text, height=400)
            
            # Highlight mentions
            if latest_response['score'] > 0:
                st.subheader("Website Mentions")
                
                # Extract mentions
                website = latest_response['website']
                clean_website = re.sub(r'^(https?://)?(www\.)?', '', website.lower())
                
                # Find mentions with context
                pattern = re.compile(r'([^.!?]*\b%s\b[^.!?]*)' % re.escape(clean_website), re.IGNORECASE)
                mentions = pattern.findall(response_text)
                
                if mentions:
                    for i, mention in enumerate(mentions):
                        st.markdown(f"**Mention {i+1}:** _{mention.strip()}_")
                else:
                    st.info("No exact matches found, but the website was detected in the response.")
        else:
            st.info("No response data available for this combination.")
    else:
        st.info("No tracking data yet. Configure your websites and keywords, then run tracking.")

with tab4:
    st.header("Export Data")
    
    if st.session_state.tracking_data:
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.tracking_data)
        
        # Filter options
        export_all = st.checkbox("Export all data", value=True)
        
        filtered_df = df.copy()
        
        if not export_all:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                exp_keywords = st.multiselect(
                    "Select Keywords:",
                    options=list(df['keyword'].unique()),
                    default=[]
                )
                if exp_keywords:
                    filtered_df = filtered_df[filtered_df['keyword'].isin(exp_keywords)]
            
            with col2:
                exp_websites = st.multiselect(
                    "Select Websites:",
                    options=list(df['website'].unique()),
                    default=[]
                )
                if exp_websites:
                    filtered_df = filtered_df[filtered_df['website'].isin(exp_websites)]
            
            with col3:
                exp_platforms = st.multiselect(
                    "Select Platforms:",
                    options=list(df['platform'].unique()),
                    default=[]
                )
                if exp_platforms:
                    filtered_df = filtered_df[filtered_df['platform'].isin(exp_platforms)]
            
            # Date range filter if available
            if 'timestamp' in filtered_df.columns:
                filtered_df['date'] = filtered_df['timestamp'].str.split(' ').str[0]
 
# Date range filter if available
            if 'timestamp' in filtered_df.columns:
                filtered_df['date'] = filtered_df['timestamp'].str.split(' ').str[0]
                
                # Convert to datetime for easier comparison
                filtered_df['date'] = pd.to_datetime(filtered_df['date'])
                
                date_range = st.date_input(
                    "Date range",
                    value=(
                        filtered_df['date'].min().date(),
                        filtered_df['date'].max().date()
                    ),
                    min_value=filtered_df['date'].min().date(),
                    max_value=filtered_df['date'].max().date()
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    mask = (filtered_df['date'] >= pd.Timestamp(start_date)) & (filtered_df['date'] <= pd.Timestamp(end_date))
                    filtered_df = filtered_df[mask]
        
        # Export options
        st.subheader("Export Format")
        export_format = st.radio(
            "Select format:",
            options=["CSV", "Excel", "JSON"]
        )
        
        include_responses = st.checkbox("Include full LLM responses", value=False)
        
        # Prepare export data
        export_df = filtered_df.copy()
        if not include_responses and 'response' in export_df.columns:
            export_df = export_df.drop(columns=['response'])
        
        # Export button
        if st.button("Generate Export"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_mention_data_{timestamp}"
            
            if export_format == "CSV":
                csv = export_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            elif export_format == "Excel":
                output = StringIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, index=False, sheet_name='LLM Mentions')
                    
                    # Add a summary sheet
                    summary_df = pd.DataFrame()
                    
                    # Website summary
                    website_summary = export_df.groupby('website')['score'].agg(['mean', 'count']).reset_index()
                    website_summary.columns = ['Website', 'Avg Score', 'Count']
                    
                    # Platform summary
                    platform_summary = export_df.groupby('platform')['score'].agg(['mean', 'count']).reset_index()
                    platform_summary.columns = ['Platform', 'Avg Score', 'Count']
                    
                    # Keyword summary
                    keyword_summary = export_df.groupby('keyword')['score'].agg(['mean', 'count']).reset_index()
                    keyword_summary.columns = ['Keyword', 'Avg Score', 'Count']
                    
                    # Write summary sheets
                    website_summary.to_excel(writer, index=False, sheet_name='Website Summary')
                    platform_summary.to_excel(writer, index=False, sheet_name='Platform Summary')
                    keyword_summary.to_excel(writer, index=False, sheet_name='Keyword Summary')
                
                b64 = base64.b64encode(output.getvalue().encode()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download Excel File</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            else:  # JSON
                json_str = export_df.to_json(orient='records')
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="{filename}.json">Download JSON File</a>'
                st.markdown(href, unsafe_allow_html=True)
                
            st.success(f"Export generated! Click the link above to download.")
            
            # Show preview
            st.subheader("Data Preview (first 10 rows)")
            st.dataframe(export_df.head(10))
    else:
        st.info("No tracking data to export. Configure your websites and keywords, then run tracking.")

# Add a settings tab
tab5 = st.tabs(["Settings"])[0]

with tab5:
    st.header("Application Settings")
    
    # Data management
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear All Tracking Data"):
            # Confirmation
            st.warning("This will delete all tracking data. Are you sure?")
            confirm = st.checkbox("Yes, I'm sure")
            
            if confirm:
                st.session_state.tracking_data = []
                # Save empty data to file
                with open('llm_mention_data.json', 'w') as f:
                    json.dump([], f)
                st.success("All tracking data cleared!")
    
    with col2:
        if st.button("Backup Data"):
            # Create backup file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"llm_mention_backup_{timestamp}.json"
            
            with open(backup_filename, 'w') as f:
                json.dump(st.session_state.tracking_data, f)
            
            # Create download link
            with open(backup_filename, 'r') as f:
                backup_data = f.read()
                
            b64 = base64.b64encode(backup_data.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="{backup_filename}">Download Backup File</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            st.success(f"Backup created! Click the link above to download.")
    
    # Restore data
    st.subheader("Restore Data")
    
    uploaded_backup = st.file_uploader("Upload backup file", type=["json"])
    if uploaded_backup is not None:
        try:
            backup_data = json.loads(uploaded_backup.getvalue().decode("utf-8"))
            
            if st.button("Restore from Backup"):
                st.session_state.tracking_data = backup_data
                
                # Save to file
                with open('llm_mention_data.json', 'w') as f:
                    json.dump(backup_data, f)
                
                st.success(f"Successfully restored {len(backup_data)} records from backup!")
        except Exception as e:
            st.error(f"Error processing backup file: {str(e)}")
    
    # Advanced settings
    st.subheader("Advanced Settings")
    
    with st.expander("Query Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            if 'query_temperature' not in st.session_state:
                st.session_state.query_temperature = 0.7
                
            temperature = st.slider(
                "LLM Temperature:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.query_temperature,
                step=0.1
            )
            
            if temperature != st.session_state.query_temperature:
                st.session_state.query_temperature = temperature
                st.success(f"Temperature updated to {temperature}")
        
        with col2:
            if 'max_tokens' not in st.session_state:
                st.session_state.max_tokens = 1000
                
            max_tokens = st.number_input(
                "Max Response Tokens:",
                min_value=100,
                max_value=8000,
                value=st.session_state.max_tokens,
                step=100
            )
            
            if max_tokens != st.session_state.max_tokens:
                st.session_state.max_tokens = max_tokens
                st.success(f"Max tokens updated to {max_tokens}")
    
    with st.expander("Scoring Algorithm"):
        if 'exact_match_score' not in st.session_state:
            st.session_state.exact_match_score = 10
            st.session_state.partial_match_score = 7
        
        col1, col2 = st.columns(2)
        
        with col1:
            exact_score = st.number_input(
                "Exact Match Score:",
                min_value=1,
                max_value=20,
                value=st.session_state.exact_match_score
            )
        
        with col2:
            partial_score = st.number_input(
                "Partial Match Score:",
                min_value=1,
                max_value=20,
                value=st.session_state.partial_match_score
            )
        
        if st.button("Update Scoring Values"):
            st.session_state.exact_match_score = exact_score
            st.session_state.partial_match_score = partial_score
            
            # Update scoring function
            def updated_check_website_mention(response, website):
                # Remove http/https and www. for more flexible matching
                clean_website = re.sub(r'^(https?://)?(www\.)?', '', website.lower())
                
                # Different mention patterns
                exact_match = clean_website.lower() in response.lower()
                partial_match = re.search(r'(\S*%s\S*)' % re.escape(clean_website), response.lower()) is not None
                
                if exact_match:
                    return st.session_state.exact_match_score  # Strong mention
                elif partial_match:
                    return st.session_state.partial_match_score  # Partial mention
                else:
                    return 0  # No mention
            
            # Replace the function in the global scope
            check_website_mention = updated_check_website_mention
            
            st.success("Scoring algorithm updated!")
            
            # Option to retroactively update scores
            if st.checkbox("Update existing scores with new algorithm"):
                if st.session_state.tracking_data:
                    updated_data = []
                    
                    for entry in st.session_state.tracking_data:
                        # Only update if it has a response
                        if 'response' in entry and 'website' in entry:
                            entry['score'] = updated_check_website_mention(entry['response'], entry['website'])
                        updated_data.append(entry)
                    
                    st.session_state.tracking_data = updated_data
                    
                    # Save updated data
                    with open('llm_mention_data.json', 'w') as f:
                        json.dump(updated_data, f)
                    
                    st.success("All existing scores have been updated!")

# Add a new tab for Prompt Testing
tab6 = st.tabs(["Prompt Testing"])[0]

with tab6:
    st.header("Prompt Testing")
    st.info("Test different prompts to see how they affect website mentions")
    
    # Prompt input
    test_prompt = st.text_area(
        "Enter your test prompt:",
        height=150,
        value="I need information about the best websites for [topic]. Please recommend some good resources."
    )
    
    # Website and platform selection
    col1, col2 = st.columns(2)
    
    with col1:
        test_website = st.selectbox(
            "Website to track:",
            options=st.session_state.websites if st.session_state.websites else ["example.com"]
        )
    
    with col2:
        test_platform = st.selectbox(
            "Platform to test:",
            options=st.session_state.platforms_to_check
        )
    
    # Test variations
    st.subheader("Prompt Variations")
    
    # Create variations
    if st.button("Generate Variations"):
        variations = [
            test_prompt,
            test_prompt + f" Have you heard of {test_website}?",
            "For the following question, please include any relevant resources such as " + test_website + ": " + test_prompt,
            "Please answer the following and mention specific websites if you know them: " + test_prompt,
            f"I'm looking for websites like {test_website} that can help with: " + test_prompt
        ]
        
        st.session_state.test_variations = variations
        
        # Display variations
        for i, var in enumerate(variations):
            st.text_area(f"Variation {i+1}", var, height=100)
        
        st.success("Variations generated! Click 'Run Test' to see results.")
    
    # Run test
    if st.button("Run Test"):
        if 'test_variations' in st.session_state and test_platform and test_website:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            total_variations = len(st.session_state.test_variations)
            
            for i, prompt in enumerate(st.session_state.test_variations):
                status_text.text(f"Testing variation {i+1}/{total_variations}")
                
                # Choose the right API function
                if test_platform == "ChatGPT":
                    response_data = query_openai(prompt, st.session_state.api_keys["openai"])
                elif test_platform == "Claude":
                    response_data = query_claude(prompt, st.session_state.api_keys["anthropic"])
                elif test_platform == "Gemini":
                    response_data = query_gemini(prompt, st.session_state.api_keys["gemini"])
                elif test_platform == "Mistral":
                    response_data = query_mistral(prompt, st.session_state.api_keys["mistral"])
                
                if not response_data["error"]:
                    # Check for website mention
                    mention_score = check_website_mention(response_data["response"], test_website)
                    
                    results.append({
                        "variation": i+1,
                        "prompt": prompt,
                        "score": mention_score,
                        "response": response_data["response"]
                    })
                else:
                    results.append({
                        "variation": i+1,
                        "prompt": prompt,
                        "score": 0,
                        "response": f"Error: {response_data['response']}"
                    })
                
                progress_bar.progress((i + 1) / total_variations)
                time.sleep(1)  # Small delay to avoid rate limits
            
            # Store results
            st.session_state.test_results = results
            
            # Display results
            st.subheader("Test Results")
            
            result_df = pd.DataFrame(results)[["variation", "score", "prompt"]]
            
            # Sort by score
            result_df = result_df.sort_values("score", ascending=False)
            
            # Display as table
            st.table(result_df)
            
            # Show the best variation
            if not result_df.empty:
                best_variation = result_df.iloc[0]
                
                st.success(f"Best variation: #{best_variation['variation']} with score {best_variation['score']}")
                
                # Show response for best variation
                best_response = next((r for r in results if r["variation"] == best_variation["variation"]), None)
                
                if best_response:
                    st.subheader("Best Variation Response")
                    st.text_area("Response Text", best_response["response"], height=300)
        else:
            st.error("Please generate variations first and ensure you have selected a website and platform.")

# Add a Help tab
tab7 = st.tabs(["Help"])[0]

with tab7:
    st.header("Help & Documentation")
    
    st.subheader("About This Tool")
    st.write("""
    The LLM Website Mention Tracker is a tool designed to help you monitor how Large Language Models (LLMs) like ChatGPT, Claude, Gemini, and Mistral mention specific websites when responding to user queries.
    
    This can be useful for:
    - Website owners tracking how often LLMs recommend their site
    - SEO professionals analyzing LLM visibility
    - Researchers studying LLM recommendation patterns
    - Content creators optimizing for AI discoverability
    """)
    
    st.subheader("How It Works")
    st.write("""
    1. You configure websites you want to track
    2. You add keywords or queries relevant to your websites
    3. The tool queries different LLM platforms with your keywords
    4. It analyzes responses to detect mentions of your websites
    5. Results are scored, tracked, and visualized on the dashboard
    """)
    
    with st.expander("Setting Up"):
        st.write("""
        **API Keys**
        - You need API keys for the LLM platforms you want to test
        - Enter these in the Configuration > API Keys section
        - Keys are stored in your session only
        
        **Websites**
        - Add the domains you want to track (e.g., example.com)
        - You can exclude the http:// or www. prefixes
        
        **Keywords**
        - These are the queries sent to the LLMs
        - Good keywords are questions or prompts a user might ask where your website could be recommended
        - You can upload keywords in bulk via CSV or TXT files
        """)
    
    with st.expander("Running Tracking"):
        st.write("""
        **Manual Tracking**
        - Select the keywords and websites for the current run
        - Click "Run Tracking Now" to start the process
        - The tool will query each selected LLM with your keywords
        - Results are saved automatically
        
        **Automated Tracking**
        - For regular tracking, set up this script with a cron job or scheduler
        - Example cron job (daily at 2 AM): `0 2 * * * cd /path/to/app && streamlit run app.py --headless`
        """)
    
    with st.expander("Analyzing Results"):
        st.write("""
        **Dashboard**
        - Shows an overview of mention performance
        - Filter by website, platform, and date
        - View average scores by keyword and platform
        - Track mention trends over time
        
        **Detailed Results**
        - Explore raw data with filtering options
        - Cross-tabulate results to compare platforms, keywords, and websites
        - Focus on specific score ranges
        
        **Responses**
        - View the full text of LLM responses
        - See highlighted mentions of your websites
        - Analyze context around mentions
        """)
    
    with st.expander("Prompt Testing"):
        st.write("""
        The Prompt Testing feature helps you optimize how you phrase queries to increase the likelihood of website mentions:
        
        1. Enter a base prompt
        2. Select the website and platform to test
        3. Generate variations of your prompt
        4. Run tests to see which variation performs best
        5. Use the insights to improve your keyword strategy
        """)
    
    st.subheader("Best Practices")
    st.markdown("""
    - **Diverse Keywords**: Test a variety of query formats and topics
    - **Regular Testing**: LLM behavior changes over time, so track regularly
    - **Compare Platforms**: Different LLMs have different recommendation patterns
    - **Optimize Content**: Use insights to improve your website's content
    - **Natural Queries**: Use questions that real users would ask
    """)
    
    st.subheader("Troubleshooting")
    with st.expander("Common Issues"):
        st.markdown("""
        **API Errors**
        - Verify your API keys are correct and have sufficient credits
        - Check that you're using the correct API versions
        - Respect rate limits by adding delays between requests
        
        **No Website Mentions**
        - Your website may not be in the LLM's training data
        - Try more specific keywords related to your website's niche
        - Ensure your website has sufficient online presence
        
        **Application Not Saving Data**
        - Check write permissions for the application directory
        - Ensure you have sufficient disk space
        """)
