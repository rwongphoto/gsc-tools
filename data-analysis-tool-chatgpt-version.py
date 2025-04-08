import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import google.generativeai as genai
import pickle
import pytz
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="SEO Performance Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1E3A8A;
        margin: 1rem 0;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
    }
    .insight-card {
        background-color: #EEF2FF;
        border-left: 5px solid #3B82F6;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Define constants
SCOPES = ['https://www.googleapis.com/auth/webmasters']
TOKEN_FILE = 'token.pickle'
CREDENTIALS_FILE = 'credentials.json'

# Initialize session state variables
if 'authorized' not in st.session_state:
    st.session_state.authorized = False
if 'gsc_service' not in st.session_state:
    st.session_state.gsc_service = None
if 'sites' not in st.session_state:
    st.session_state.sites = []
if 'selected_site' not in st.session_state:
    st.session_state.selected_site = None
if 'date_range' not in st.session_state:
    st.session_state.date_range = (datetime.now() - timedelta(days=28)).strftime('%Y-%m-%d')
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.now().strftime('%Y-%m-%d')
if 'gemini_initialized' not in st.session_state:
    st.session_state.gemini_initialized = False
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None
if 'query_data' not in st.session_state:
    st.session_state.query_data = None
if 'page_data' not in st.session_state:
    st.session_state.page_data = None
if 'trend_analysis' not in st.session_state:
    st.session_state.trend_analysis = None
if 'performance_summary' not in st.session_state:
    st.session_state.performance_summary = None

# Function to authenticate and build Google Search Console service
def authenticate_gsc():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                st.error(f"Error refreshing credentials: {str(e)}")
                return False
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                st.error("Credentials file not found! Please upload your Google API credentials.")
                return False
            try:
                with open(CREDENTIALS_FILE, 'r') as f:
                    client_config = json.load(f)
                
                # Validate expected keys for Desktop application OAuth
                if 'web' not in client_config and 'installed' not in client_config:
                    st.error("Invalid credentials format. Please ensure your credentials are for a Desktop application.")
                    return False
            except Exception as e:
                st.error(f"Error reading credentials file: {str(e)}")
                return False
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_FILE, 
                    SCOPES,
                    redirect_uri='urn:ietf:wg:oauth:2.0:oob'
                )
                auth_url, _ = flow.authorization_url(
                    access_type='offline',
                    include_granted_scopes='true'
                )
                st.markdown("### Google Authentication Required")
                st.markdown(f"[Click here to authorize]({auth_url})")
                st.markdown("Follow the instructions and copy the authorization code below.")
                auth_code = st.text_input("Enter the authorization code:", type="password")
                if auth_code:
                    flow.fetch_token(code=auth_code)
                    creds = flow.credentials
                    with open(TOKEN_FILE, 'wb') as token:
                        pickle.dump(creds, token)
                    st.success("Authentication successful!")
                else:
                    st.info("Please complete the authorization steps above.")
                    return False
            except Exception as e:
                st.error(f"Authentication failed: {str(e)}")
                return False

    try:
        service = build('searchconsole', 'v1', credentials=creds)
        st.session_state.gsc_service = service
        st.session_state.authorized = True
        return True
    except Exception as e:
        st.error(f"Failed to build GSC service: {str(e)}")
        return False

# Function to initialize Gemini API
def initialize_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="models/gemini-2.5-pro",
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
        st.session_state.gemini_model = model
        st.session_state.gemini_initialized = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        return False

# Function to fetch sites from Google Search Console
def get_sites():
    if not st.session_state.authorized:
        return []
    try:
        sites = st.session_state.gsc_service.sites().list().execute()
        site_list = []
        if 'siteEntry' in sites:
            for site in sites['siteEntry']:
                site_list.append(site['siteUrl'])
        st.session_state.sites = site_list
        return site_list
    except Exception as e:
        st.error(f"Failed to fetch sites: {str(e)}")
        return []

# Function to fetch query data
def get_query_data(site_url, start_date, end_date, row_limit=5000):
    if not st.session_state.authorized:
        return None
    request = {
        'startDate': start_date,
        'endDate': end_date,
        'dimensions': ['query', 'date'],
        'rowLimit': row_limit,
        'aggregationType': 'auto'
    }
    try:
        response = st.session_state.gsc_service.searchanalytics().query(siteUrl=site_url, body=request).execute()
        if 'rows' not in response:
            st.warning("No query data found for the selected date range.")
            return None
        data = []
        for row in response['rows']:
            data.append({
                'query': row['keys'][0],
                'date': row['keys'][1],
                'clicks': row.get('clicks', 0),
                'impressions': row.get('impressions', 0),
                'ctr': row.get('ctr', 0),
                'position': row.get('position', 0)
            })
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df['ctr'] = df['ctr'] * 100  # Convert to percentage
        return df
    except Exception as e:
        st.error(f"Failed to fetch query data: {str(e)}")
        return None

# Function to fetch page data
def get_page_data(site_url, start_date, end_date, row_limit=5000):
    if not st.session_state.authorized:
        return None
    request = {
        'startDate': start_date,
        'endDate': end_date,
        'dimensions': ['page', 'date'],
        'rowLimit': row_limit,
        'aggregationType': 'auto'
    }
    try:
        response = st.session_state.gsc_service.searchanalytics().query(siteUrl=site_url, body=request).execute()
        if 'rows' not in response:
            st.warning("No page data found for the selected date range.")
            return None
        data = []
        for row in response['rows']:
            data.append({
                'page': row['keys'][0],
                'date': row['keys'][1],
                'clicks': row.get('clicks', 0),
                'impressions': row.get('impressions', 0),
                'ctr': row.get('ctr', 0),
                'position': row.get('position', 0)
            })
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df['ctr'] = df['ctr'] * 100  # Convert to percentage
        return df
    except Exception as e:
        st.error(f"Failed to fetch page data: {str(e)}")
        return None

# Function to generate trend analysis and performance summary using Gemini
def generate_trend_analysis(query_df, page_df, site_url):
    if not st.session_state.gemini_initialized or not st.session_state.gemini_model:
        st.error("Gemini API not initialized!")
        return None
    try:
        # Prepare query and page summaries
        query_summary = query_df.groupby('query').agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'position': 'mean'
        }).reset_index().sort_values('clicks', ascending=False).head(20)
        
        page_summary = page_df.groupby('page').agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'position': 'mean'
        }).reset_index().sort_values('clicks', ascending=False).head(20)
        
        # Prepare weekly trends
        query_trends = query_df.groupby(['query', pd.Grouper(key='date', freq='W-MON')]).agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'position': 'mean'
        }).reset_index()
        
        # Create prompt for analysis
        prompt = f"""
Analyze the following SEO data from Google Search Console for {site_url}:

Top 20 queries by clicks:
{query_summary.to_string()}

Top 20 pages by clicks:
{page_summary.to_string()}

Based on this data, provide a comprehensive SEO analysis including:
1. Key performance trends and patterns
2. Top performing queries and pages
3. Opportunities for optimization
4. Content recommendations based on search patterns
5. Specific areas for improvement

Format your response in clear sections with bullet points for easy readability.
        """
        response = st.session_state.gemini_model.generate_content(prompt)
        
        # Create prompt for performance summary
        summary_prompt = f"""
Based on the Google Search Console data for {site_url}, provide 5 key performance metrics and insights about:
1. Overall search visibility trends
2. Click-through rate performance
3. Ranking position changes
4. Top query performance
5. Page performance patterns

Format each insight as a brief bullet point (1-2 sentences max).
        """
        summary_response = st.session_state.gemini_model.generate_content(summary_prompt)
        
        return {
            'full_analysis': response.text,
            'summary': summary_response.text
        }
    except Exception as e:
        st.error(f"Failed to generate analysis with Gemini: {str(e)}")
        return None

# Function to upload credentials via file uploader
def upload_credentials():
    uploaded_file = st.file_uploader("Upload your Google API credentials JSON file", type="json")
    use_existing_token = st.checkbox("Use existing token file (if available)")
    if use_existing_token and os.path.exists(TOKEN_FILE):
        st.success("Using existing authentication token.")
        return True
    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue().decode('utf-8')
            client_config = json.loads(content)
            if 'web' not in client_config and 'installed' not in client_config:
                st.error("Invalid credentials format. Please ensure your credentials are for a Desktop application.")
                return False
            with open(CREDENTIALS_FILE, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("Credentials file uploaded successfully!")
            return True
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid credentials file.")
            return False
        except Exception as e:
            st.error(f"Error saving credentials: {str(e)}")
            return False
    return False

# Function to upload an existing token file
def upload_token():
    uploaded_token = st.file_uploader("Upload an existing token.pickle file", type=["pickle"])
    if uploaded_token is not None:
        try:
            with open(TOKEN_FILE, "wb") as f:
                f.write(uploaded_token.getbuffer())
            st.success("Token file uploaded successfully!")
            return True
        except Exception as e:
            st.error(f"Error saving token: {str(e)}")
            return False
    return False

# --------------------------
# App UI and Sidebar Setup
# --------------------------

st.markdown('<h1 class="main-header">SEO Performance Analyzer</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("### Authentication")
    
    # GSC Authentication
    if not st.session_state.authorized:
        st.info("Please upload your Google Search Console API credentials to begin.")
        if upload_credentials():
            if st.button("Authenticate with Google Search Console"):
                with st.spinner("Authenticating..."):
                    if authenticate_gsc():
                        st.success("Authentication successful!")
                        get_sites()
    else:
        st.success("✅ Connected to Google Search Console")
        if st.button("Refresh Sites"):
            with st.spinner("Refreshing sites..."):
                get_sites()
    
    # Gemini API Setup
    st.markdown("### Gemini API Setup")
    gemini_api_key = st.text_input("Enter your Gemini API Key", type="password")
    if gemini_api_key and not st.session_state.gemini_initialized:
        if st.button("Initialize Gemini API"):
            with st.spinner("Initializing Gemini API..."):
                if initialize_gemini(gemini_api_key):
                    st.success("Gemini API initialized successfully!")
    elif st.session_state.gemini_initialized:
        st.success("✅ Gemini API initialized")
    
    # Site Selection
    if st.session_state.authorized and st.session_state.sites:
        st.markdown("### Site Selection")
        selected_site = st.selectbox("Select a website", st.session_state.sites)
        if selected_site:
            st.session_state.selected_site = selected_site

    # Date Range Selection
    if st.session_state.selected_site:
        st.markdown("### Date Range")
        date_options = {
            "Last 7 days": 7,
            "Last 28 days": 28,
            "Last 3 months": 90,
            "Last 6 months": 180,
            "Last 12 months": 365,
            "Custom Range": 0
        }
        selected_date_option = st.selectbox("Select date range", list(date_options.keys()))
        if selected_date_option == "Custom Range":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=28), max_value=datetime.now())
            with col2:
                end_date = st.date_input("End Date", value=datetime.now(), max_value=datetime.now())
            st.session_state.date_range = start_date.strftime('%Y-%m-%d')
            st.session_state.end_date = end_date.strftime('%Y-%m-%d')
        else:
            days = date_options[selected_date_option]
            st.session_state.date_range = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            st.session_state.end_date = datetime.now().strftime('%Y-%m-%d')
        
        if st.button("Fetch Data & Analyze"):
            with st.spinner("Fetching data from Google Search Console..."):
                st.session_state.query_data = get_query_data(
                    st.session_state.selected_site,
                    st.session_state.date_range,
                    st.session_state.end_date
                )
                st.session_state.page_data = get_page_data(
                    st.session_state.selected_site,
                    st.session_state.date_range,
                    st.session_state.end_date
                )
            if st.session_state.query_data is not None and st.session_state.page_data is not None:
                st.success("Data fetched successfully!")
                if st.session_state.gemini_initialized:
                    with st.spinner("Generating AI analysis with Gemini..."):
                        analysis = generate_trend_analysis(
                            st.session_state.query_data,
                            st.session_state.page_data,
                            st.session_state.selected_site
                        )
                        if analysis:
                            st.session_state.trend_analysis = analysis['full_analysis']
                            st.session_state.performance_summary = analysis['summary']
                            st.success("Analysis completed!")

# --------------------------
# Main Content and Dashboard
# --------------------------
if not st.session_state.authorized:
    st.info("Please authenticate with Google Search Console via the sidebar to begin.")
elif not st.session_state.selected_site:
    st.info("Please select a website in the sidebar to analyze.")
elif st.session_state.query_data is None or st.session_state.page_data is None:
    st.info("Click 'Fetch Data & Analyze' in the sidebar to load your SEO data.")
else:
    st.markdown('<h2 class="sub-header">SEO Performance Dashboard</h2>', unsafe_allow_html=True)
    st.markdown(f"Data from **{st.session_state.date_range}** to **{st.session_state.end_date}**")
    
    query_df = st.session_state.query_data
    page_df = st.session_state.page_data

    total_clicks = int(query_df['clicks'].sum())
    total_impressions = int(query_df['impressions'].sum())
    avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    avg_position = query_df['position'].mean()
    
    # Overview Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{total_clicks:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Clicks</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{total_impressions:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Impressions</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{avg_ctr:.2f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average CTR</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{avg_position:.1f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average Position</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Daily Metrics Aggregation and Charting
    daily_agg = query_df.groupby('date').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean'
    }).reset_index()
    daily_agg['ctr'] = daily_agg.apply(
        lambda row: (row['clicks'] / row['impressions'] * 100) if row['impressions'] > 0 else 0,
        axis=1
    )
    
    tab1, tab2, tab3 = st.tabs(["Clicks & Impressions", "CTR", "Position"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_agg['date'],
            y=daily_agg['clicks'],
            mode='lines',
            name='Clicks',
            line=dict(width=3)
        ))
        fig.add_trace(go.Scatter(
            x=daily_agg['date'],
            y=daily_agg['impressions'],
            mode='lines',
            name='Impressions',
            line=dict(width=3),
            yaxis='y2'
        ))
        fig.update_layout(
            title="Clicks & Impressions Over Time",
            xaxis_title="Date",
            yaxis=dict(title="Clicks"),
            yaxis2=dict(
                title="Impressions",
                anchor="x",
                overlaying="y",
                side="right"
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = px.line(
            daily_agg,
            x='date', 
            y='ctr', 
            title="Click-Through Rate Over Time",
            labels={'date': 'Date', 'ctr': 'CTR (%)'},
            line_shape='spline'
        )
        fig.update_traces(line=dict(width=3))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        fig = px.line(
            daily_agg,
            x='date', 
            y='position', 
            title="Average Position Over Time",
            labels={'date': 'Date', 'position': 'Position'},
            line_shape='spline'
        )
        fig.update_traces(line=dict(width=3))
        fig.update_layout(
            height=500,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Queries and Top Pages Aggregations
    # Top Queries: Compute weighted CTR from group data
    query_group = query_df.groupby('query').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean'
    }).reset_index()
    query_group['ctr'] = query_group.apply(
        lambda row: (row['clicks'] / row['impressions'] * 100) if row['impressions'] > 0 else 0,
        axis=1
    )
    top_queries = query_group.sort_values('clicks', ascending=False).head(10)
    
    # Top Pages: Compute weighted CTR from group data
    page_group = page_df.groupby('page').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean'
    }).reset_index()
    page_group['ctr'] = page_group.apply(
        lambda row: (row['clicks'] / row['impressions'] * 100) if row['impressions'] > 0 else 0,
        axis=1
    )
    top_pages = page_group.sort_values('clicks', ascending=False).head(10)
    top_pages['page_path'] = top_pages['page'].apply(lambda x: x.replace(st.session_state.selected_site, ''))
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<h3 class="sub-header">Top Queries</h3>', unsafe_allow_html=True)
        fig = px.bar(
            top_queries,
            x='query',
            y='clicks',
            hover_data=['impressions', 'ctr', 'position'],
            labels={'query': 'Query', 'clicks': 'Clicks'},
            title="Top 10 Queries by Clicks",
            color='clicks',
            color_continuous_scale=px.colors.sequential.Blues
        )
        fig.update_layout(height=500)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        st.markdown('<h3 class="sub-header">Top Pages</h3>', unsafe_allow_html=True)
        fig = px.bar(
            top_pages,
            x='page_path',
            y='clicks',
            hover_data=['impressions', 'ctr', 'position'],
            labels={'page_path': 'Page Path', 'clicks': 'Clicks'},
            title="Top 10 Pages by Clicks",
            color='clicks',
            color_continuous_scale=px.colors.sequential.Greens
        )
        fig.update_layout(height=500)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Gemini Analysis Section
    if st.session_state.gemini_initialized and st.session_state.trend_analysis:
        st.markdown('<h2 class="sub-header">Gemini AI Analysis</h2>', unsafe_allow_html=True)
        if st.session_state.performance_summary:
            st.markdown("### Key Insights")
            summary_lines = st.session_state.performance_summary.strip().split('\n')
            for line in summary_lines:
                clean_line = line.strip().replace('- ', '').replace('* ', '')
                if clean_line:
                    st.markdown(f'<div class="insight-card">{clean_line}</div>', unsafe_allow_html=True)
        with st.expander("View Full Analysis", expanded=False):
            st.markdown(st.session_state.trend_analysis)
        st.markdown('<h3 class="sub-header">Action Recommendations</h3>', unsafe_allow_html=True)
        recommendation_prompt = f"""
Based on the SEO data for {st.session_state.selected_site}, provide 5 specific, actionable recommendations to improve SEO performance.

Format your response as 5 numbered bullet points. Each bullet point should start with an action verb and be concise (1-2 sentences max).
        """
        with st.spinner("Generating recommendations..."):
            try:
                recommendations = st.session_state.gemini_model.generate_content(recommendation_prompt)
                st.markdown(recommendations.text)
            except Exception as e:
                st.error(f"Failed to generate recommendations: {str(e)}")

if __name__ == "__main__":
    pass
