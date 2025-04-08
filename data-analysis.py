import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Removed as it's not used
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json
from google.oauth2 import service_account
# from google.oauth2.credentials import Credentials # Not explicitly used after service_account import
from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow # Not used in service account flow
# from google.auth.transport.requests import Request # Not explicitly used after service_account import
import google.generativeai as genai
# import pickle # Not used in service account flow
# import pytz # Not used
# from pathlib import Path # Not used

# Set page config
st.set_page_config(
    page_title="SEO Performance Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A; /* Dark Blue */
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1E3A8A; /* Dark Blue */
        margin: 1rem 0;
    }
    .card {
        background-color: #ffffff; /* White background */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); /* Softer shadow */
        margin-bottom: 20px;
        border: 1px solid #e5e7eb; /* Light border */
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A; /* Dark Blue */
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280; /* Gray */
    }
    .insight-card {
        background-color: #EFF6FF; /* Lighter Blue */
        border-left: 5px solid #3B82F6; /* Blue accent */
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
        color: #1F2937; /* Darker text */
    }
    /* Improve sidebar readability */
    .stSidebar .stButton>button {
        width: 100%;
    }
    .stSidebar .stSelectbox, .stSidebar .stDateInput {
        margin-bottom: 10px;
    }
    /* Improve tab styling */
    .stTabs [data-baseweb="tab-list"] {
		gap: 12px; /* Space between tabs */
	}
    .stTabs [data-baseweb="tab"] {
		height: 40px;
        white-space: pre-wrap;
		background-color: #F3F4F6; /* Light Gray background */
		border-radius: 8px 8px 0px 0px;
		gap: 4px;
        padding: 10px 15px;
        transition: background-color 0.2s ease;
	}
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE; /* Light Blue for selected tab */
        color: #1E40AF; /* Darker Blue text */
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Define constants
SCOPES = ['https://www.googleapis.com/auth/webmasters']
# TOKEN_FILE = 'token.pickle' # Not needed for service account
# CREDENTIALS_FILE = 'credentials.json' # Not needed when using secrets

# Initialize session state (using dictionary update for brevity)
st.session_state.update({
    'authorized': st.session_state.get('authorized', False),
    'gsc_service': st.session_state.get('gsc_service', None),
    'sites': st.session_state.get('sites', []),
    'selected_site': st.session_state.get('selected_site', None),
    'date_range': st.session_state.get('date_range', (datetime.now() - timedelta(days=28)).strftime('%Y-%m-%d')),
    'end_date': st.session_state.get('end_date', datetime.now().strftime('%Y-%m-%d')),
    'gemini_initialized': st.session_state.get('gemini_initialized', False),
    'gemini_model': st.session_state.get('gemini_model', None),
    'query_data': st.session_state.get('query_data', None),
    'page_data': st.session_state.get('page_data', None),
    'trend_analysis': st.session_state.get('trend_analysis', None),
    'performance_summary': st.session_state.get('performance_summary', None)
})

# Authentication function using Streamlit Secrets
def authenticate_gsc():
    """Authenticates with GSC using service account credentials from Streamlit Secrets."""
    try:
        # Check if credentials exist in secrets
        if "gcp_credentials" not in st.secrets:
            st.error("GCP service account credentials not found in Streamlit Secrets (key: 'gcp_credentials').")
            st.caption("Please ensure your service account JSON key content is stored under `[gcp_credentials]` in your `secrets.toml` file.")
            return False

        # Create credentials object from secrets
        credentials_dict = dict(st.secrets["gcp_credentials"])

        # Create credentials object
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=SCOPES
        )

        # Build the service
        service = build('searchconsole', 'v1', credentials=credentials)
        st.session_state.gsc_service = service
        st.session_state.authorized = True
        return True

    except json.JSONDecodeError:
        st.error("Invalid format for GCP credentials in Streamlit Secrets. Please ensure it's valid JSON.")
        return False
    except Exception as e:
        st.error(f"GSC Authentication failed: {str(e)}")
        return False

# Initialize Gemini API using Streamlit Secrets
def initialize_gemini():
    """Initializes the Gemini API using the API key from Streamlit Secrets."""
    try:
        # Check if Gemini API key exists in secrets
        if "gemini" not in st.secrets or "api_key" not in st.secrets["gemini"]:
            st.error("Gemini API key not found in Streamlit Secrets (key: 'gemini.api_key').")
            st.caption("Please ensure your Gemini API key is stored under `[gemini]` with key `api_key` in your `secrets.toml` file.")
            return False

        api_key = st.secrets["gemini"]["api_key"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            # Use the more powerful gemini-1.5-pro-latest model
            model_name="gemini-1.5-pro-latest", # <--- CORRECTED MODEL NAME (Alternative)
            generation_config={
                "temperature": 0.3,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
            # safety_settings=[...]
        )
        st.session_state.gemini_model = model
        st.session_state.gemini_initialized = True
        return True

    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        return False

    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        return False

# Function to fetch sites from GSC
@st.cache_data(ttl=3600) # Cache site list for 1 hour
def get_sites(_gsc_service):
    """Fetches the list of sites associated with the authenticated GSC account."""
    if not st.session_state.authorized or not _gsc_service:
        return []

    try:
        sites_response = _gsc_service.sites().list().execute()
        site_list = [site['siteUrl'] for site in sites_response.get('siteEntry', [])]
        st.session_state.sites = site_list
        return site_list
    except Exception as e:
        st.error(f"Failed to fetch sites: {str(e)}")
        return []

# Function to fetch query data from GSC
@st.cache_data(ttl=900) # Cache data for 15 minutes
def get_gsc_data(_gsc_service, site_url, start_date, end_date, dimensions, row_limit=5000):
    """Fetches search analytics data from GSC for given dimensions."""
    if not st.session_state.authorized or not _gsc_service:
        st.warning("GSC not authorized.")
        return None

    request = {
        'startDate': start_date,
        'endDate': end_date,
        'dimensions': dimensions,
        'rowLimit': row_limit,
        'aggregationType': 'auto' # Let GSC decide aggregation
    }

    try:
        response = _gsc_service.searchanalytics().query(siteUrl=site_url, body=request).execute()

        if 'rows' not in response or not response['rows']:
            st.warning(f"No data found for dimensions {', '.join(dimensions)} in the selected date range.")
            return pd.DataFrame() # Return empty DataFrame

        # Convert to DataFrame
        data = []
        num_keys = len(dimensions)
        for row in response['rows']:
            row_data = {dimensions[i]: row['keys'][i] for i in range(num_keys)}
            row_data.update({
                'clicks': row.get('clicks', 0),
                'impressions': row.get('impressions', 0),
                'ctr': row.get('ctr', 0) * 100, # Convert to percentage immediately
                'position': row.get('position', 0)
            })
            data.append(row_data)

        df = pd.DataFrame(data)

        # Convert date column if present
        if 'date' in dimensions:
            df['date'] = pd.to_datetime(df['date'])

        return df

    except Exception as e:
        st.error(f"Failed to fetch {', '.join(dimensions)} data: {str(e)}")
        return None


# Function to generate trend analysis with Gemini
# Consider caching this if analysis is expensive and data doesn't change rapidly
# @st.cache_data(ttl=1800) # Cache analysis for 30 mins
def generate_trend_analysis(_gemini_model, query_df, page_df, site_url):
    """Generates SEO analysis and summary using Gemini."""
    if not st.session_state.gemini_initialized or not _gemini_model:
        st.error("Gemini API not initialized!")
        return None

    if query_df is None or query_df.empty or page_df is None or page_df.empty:
        st.warning("Cannot generate analysis with empty data.")
        return None

    try:
        # --- Prepare Data Summaries ---
        # Ensure position is numeric and handle potential errors
        query_df['position'] = pd.to_numeric(query_df['position'], errors='coerce')
        page_df['position'] = pd.to_numeric(page_df['position'], errors='coerce')

        # Aggregate Query Data
        query_summary = query_df.groupby('query').agg(
            total_clicks=('clicks', 'sum'),
            total_impressions=('impressions', 'sum'),
            average_ctr=('ctr', 'mean'), # Simple average CTR across days
            average_position=('position', 'mean')
        ).sort_values('total_clicks', ascending=False).head(20).reset_index()
        query_summary = query_summary.round(2) # Round for cleaner output

        # Aggregate Page Data
        page_summary = page_df.groupby('page').agg(
            total_clicks=('clicks', 'sum'),
            total_impressions=('impressions', 'sum'),
            average_ctr=('ctr', 'mean'),
            average_position=('position', 'mean')
        ).sort_values('total_clicks', ascending=False).head(20).reset_index()
        page_summary = page_summary.round(2)

        # Overall Performance Metrics
        start_date = query_df['date'].min().strftime('%Y-%m-%d')
        end_date = query_df['date'].max().strftime('%Y-%m-%d')
        total_clicks = query_df['clicks'].sum()
        total_impressions = query_df['impressions'].sum()
        overall_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        overall_avg_position = query_df['position'].mean()

        # --- Generate Full Analysis Prompt ---
        prompt = f"""
        Analyze the Google Search Console data for the website '{site_url}' covering the period from {start_date} to {end_date}.

        Overall Performance Metrics:
        - Total Clicks: {total_clicks:,}
        - Total Impressions: {total_impressions:,}
        - Average CTR: {overall_ctr:.2f}%
        - Average Position: {overall_avg_position:.2f}

        Top 20 Queries by Clicks:
        ```
        {query_summary.to_string(index=False)}
        ```

        Top 20 Pages by Clicks:
        ```
        {page_summary.to_string(index=False)}
        ```

        Based ONLY on the provided data, generate a concise SEO analysis focusing on:
        1.  **Key Performance Summary:** Briefly state the overall clicks, impressions, CTR, and position trends (improving, declining, stable?).
        2.  **Top Performers:** Identify the 2-3 most significant queries and pages driving traffic. Mention their clicks and average position.
        3.  **Potential Opportunities:** Highlight 1-2 queries or pages with high impressions but low CTR or poor position that could be optimized.
        4.  **Areas for Investigation:** Suggest 1-2 patterns or anomalies in the data that warrant further investigation (e.g., a sudden drop/increase, specific query behavior).

        **Instructions:**
        - Be concise and data-driven. Refer directly to the metrics provided.
        - Use clear headings for each section (e.g., ## Key Performance Summary).
        - Use bullet points for lists.
        - Do NOT invent information or make recommendations beyond what the data suggests. Focus on *analysis* of the provided numbers.
        - Keep the analysis brief and focused on the most impactful findings.
        """

        # Generate Full Analysis
        response = _gemini_model.generate_content(prompt)

        # --- Generate Performance Summary Prompt ---
        summary_prompt = f"""
        Based *only* on the following Google Search Console metrics for '{site_url}' ({start_date} to {end_date}):
        - Total Clicks: {total_clicks:,}
        - Total Impressions: {total_impressions:,}
        - Average CTR: {overall_ctr:.2f}%
        - Average Position: {overall_avg_position:.2f}
        - Top query by clicks: '{query_summary.iloc[0]['query']}' ({query_summary.iloc[0]['total_clicks']:,} clicks)
        - Top page by clicks: '{page_summary.iloc[0]['page']}' ({page_summary.iloc[0]['total_clicks']:,} clicks)

        Provide 5 brief, key insights (1 sentence each) summarizing the most important performance aspects. Start each insight with a bullet point (*). Focus on stating facts from the data.

        Example format:
        * Total clicks reached X, driven primarily by [top query/page].
        * Overall average position is Y, indicating [strong/weak] average visibility.
        """

        # Generate Summary
        summary_response = _gemini_model.generate_content(summary_prompt)

        return {
            'full_analysis': response.text,
            'summary': summary_response.text
        }
    except Exception as e:
        st.error(f"Failed to generate analysis with Gemini: {str(e)}")
        return None

# --- Sidebar UI ---
with st.sidebar:
    st.markdown("## Configuration")
    st.divider()

    # GSC Authentication
    st.markdown("### Google Search Console")
    if not st.session_state.authorized:
        st.info("Connect to Google Search Console to load sites.")
        if st.button("🔗 Connect to GSC", key="connect_gsc"):
            with st.spinner("Authenticating..."):
                if authenticate_gsc():
                    st.success("Authentication successful!")
                    # Trigger site fetching after successful auth
                    st.session_state.sites = get_sites(st.session_state.gsc_service)
                    st.rerun() # Rerun to update sidebar state
                else:
                    st.error("Authentication failed. Check secrets and permissions.")
    else:
        st.success("✅ Connected to GSC")
        # Fetch sites if not already fetched or if refresh is clicked
        if not st.session_state.sites:
             st.session_state.sites = get_sites(st.session_state.gsc_service)

        if st.button("🔄 Refresh Sites", key="refresh_sites"):
            with st.spinner("Refreshing sites..."):
                st.session_state.sites = get_sites(st.session_state.gsc_service)
                st.success("Site list refreshed.")
                st.rerun()


    # Gemini API Initialization
    st.markdown("### Gemini AI Analysis")
    if not st.session_state.gemini_initialized:
        st.info("Initialize Gemini to enable AI analysis.")
        if st.button("✨ Initialize Gemini API", key="init_gemini"):
            with st.spinner("Initializing Gemini API..."):
                if initialize_gemini():
                    st.success("Gemini API initialized successfully!")
                    st.rerun()
                else:
                    st.error("Gemini initialization failed. Check API key in secrets.")
    else:
        st.success("✅ Gemini API initialized")

    st.divider()

    # Site Selection
    if st.session_state.authorized and st.session_state.sites:
        st.markdown("### Site Selection")
        # Use index to handle potential site list updates gracefully
        site_options = st.session_state.sites
        selected_site_index = site_options.index(st.session_state.selected_site) if st.session_state.selected_site in site_options else 0

        selected_site = st.selectbox(
            "Select a website",
            options=site_options,
            index=selected_site_index,
            key="site_selector"
        )
        # Update session state only if selection changes
        if selected_site and selected_site != st.session_state.selected_site:
            st.session_state.selected_site = selected_site
            # Clear old data when site changes
            st.session_state.query_data = None
            st.session_state.page_data = None
            st.session_state.trend_analysis = None
            st.session_state.performance_summary = None
            st.rerun()

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

        selected_date_option = st.selectbox(
            "Select date range",
            list(date_options.keys()),
            index=1, # Default to "Last 28 days"
            key="date_option_selector"
            )

        start_date_val = datetime.strptime(st.session_state.date_range, '%Y-%m-%d').date()
        end_date_val = datetime.strptime(st.session_state.end_date, '%Y-%m-%d').date()

        if selected_date_option == "Custom Range":
            # Ensure dates are in correct order and within limits
            today = datetime.now().date()
            start_date_input = st.date_input("Start Date",
                                            value=start_date_val,
                                            max_value=today,
                                            key="start_date")
            end_date_input = st.date_input("End Date",
                                          value=end_date_val,
                                          min_value=start_date_input,
                                          max_value=today,
                                          key="end_date")

            # Update session state if dates change
            new_start_date_str = start_date_input.strftime('%Y-%m-%d')
            new_end_date_str = end_date_input.strftime('%Y-%m-%d')
            if new_start_date_str != st.session_state.date_range or new_end_date_str != st.session_state.end_date:
                st.session_state.date_range = new_start_date_str
                st.session_state.end_date = new_end_date_str
                # Clear old data when date range changes
                st.session_state.query_data = None
                st.session_state.page_data = None
                st.session_state.trend_analysis = None
                st.session_state.performance_summary = None
                st.rerun()

        else:
            days = date_options[selected_date_option]
            new_start_date = (datetime.now() - timedelta(days=days)).date()
            new_end_date = datetime.now().date()
            new_start_date_str = new_start_date.strftime('%Y-%m-%d')
            new_end_date_str = new_end_date.strftime('%Y-%m-%d')

            # Update session state if dates change
            if new_start_date_str != st.session_state.date_range or new_end_date_str != st.session_state.end_date:
                st.session_state.date_range = new_start_date_str
                st.session_state.end_date = new_end_date_str
                 # Clear old data when date range changes
                st.session_state.query_data = None
                st.session_state.page_data = None
                st.session_state.trend_analysis = None
                st.session_state.performance_summary = None
                st.rerun()
            # Display the calculated dates for non-custom ranges
            st.caption(f"Dates: {st.session_state.date_range} to {st.session_state.end_date}")


        # Fetch data button
        st.divider()
        if st.button("🚀 Fetch Data & Analyze", key="fetch_analyze", use_container_width=True, type="primary"):
            if not st.session_state.selected_site:
                st.warning("Please select a site first.")
            else:
                data_loaded = False
                with st.spinner("Fetching data from Google Search Console..."):
                    # Fetch query data
                    query_df = get_gsc_data(
                        st.session_state.gsc_service,
                        st.session_state.selected_site,
                        st.session_state.date_range,
                        st.session_state.end_date,
                        ['query', 'date'] # Dimensions
                    )
                    st.session_state.query_data = query_df

                    # Fetch page data
                    page_df = get_gsc_data(
                        st.session_state.gsc_service,
                        st.session_state.selected_site,
                        st.session_state.date_range,
                        st.session_state.end_date,
                        ['page', 'date'] # Dimensions
                    )
                    st.session_state.page_data = page_df

                # Check if data fetching was successful (even if empty)
                if query_df is not None and page_df is not None:
                    st.success("Data fetched successfully!")
                    data_loaded = True
                else:
                    st.error("Failed to fetch some or all data.")

                # Proceed to analysis only if data is loaded and Gemini is ready
                if data_loaded and st.session_state.gemini_initialized and not query_df.empty and not page_df.empty:
                    with st.spinner("Generating AI analysis with Gemini..."):
                        analysis = generate_trend_analysis(
                            st.session_state.gemini_model,
                            st.session_state.query_data,
                            st.session_state.page_data,
                            st.session_state.selected_site
                        )

                        if analysis:
                            st.session_state.trend_analysis = analysis['full_analysis']
                            st.session_state.performance_summary = analysis['summary']
                            st.success("Analysis completed!")
                        else:
                            st.warning("AI analysis could not be generated.")
                elif data_loaded and (query_df.empty or page_df.empty):
                     st.info("No data found for the selected period. AI analysis skipped.")
                elif data_loaded and not st.session_state.gemini_initialized:
                    st.info("Gemini not initialized. Skipping AI analysis.")


# --- Main Content Area ---
st.markdown('<h1 class="main-header">SEO Performance Analyzer</h1>', unsafe_allow_html=True)

# Display guidance messages
if not st.session_state.authorized:
    st.info("👈 Please connect to Google Search Console in the sidebar to get started.")
elif not st.session_state.selected_site:
    st.info("👈 Please select a website in the sidebar to analyze.")
elif st.session_state.query_data is None or st.session_state.page_data is None:
    st.info("👈 Click 'Fetch Data & Analyze' in the sidebar to load your SEO data.")
# Display dashboard only if data is loaded (can be empty dataframes)
elif st.session_state.query_data is not None and st.session_state.page_data is not None:

    query_df = st.session_state.query_data
    page_df = st.session_state.page_data

    # Check if dataframes are empty
    if query_df.empty and page_df.empty:
        st.warning(f"No Google Search Console data found for **{st.session_state.selected_site}** between **{st.session_state.date_range}** and **{st.session_state.end_date}**. Please select a different date range or site.")
    else:
        # --- Dashboard Content ---
        st.markdown(f'<h2 class="sub-header">Performance Dashboard for {st.session_state.selected_site}</h2>', unsafe_allow_html=True)
        st.caption(f"Data from **{st.session_state.date_range}** to **{st.session_state.end_date}**")
        st.divider()

        # --- Overview Metrics ---
        st.markdown('### Overview Metrics')
        # Ensure calculations handle potential division by zero if impressions are zero
        total_clicks = int(query_df['clicks'].sum())
        total_impressions = int(query_df['impressions'].sum())
        avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        # Calculate weighted average position if impressions > 0, otherwise simple mean or 0
        if total_impressions > 0:
             # Ensure position is numeric, coerce errors to NaN, then drop NaNs for calculation
             query_df['position_numeric'] = pd.to_numeric(query_df['position'], errors='coerce')
             weighted_pos = (query_df['position_numeric'] * query_df['impressions']).sum()
             avg_position = weighted_pos / total_impressions if total_impressions > 0 else query_df['position_numeric'].mean()
             avg_position = avg_position if pd.notna(avg_position) else 0 # Handle case where all positions might be NaN
        else:
             avg_position = 0

        # Metrics row using cards
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

        st.divider()

        # --- Trend Charts ---
        st.markdown('### Performance Trends Over Time')

        # Aggregate daily metrics
        if not query_df.empty:
            # Calculate weighted daily CTR and Position
            daily_metrics = query_df.groupby('date').agg(
                daily_clicks=('clicks', 'sum'),
                daily_impressions=('impressions', 'sum')
            ).reset_index()

            # Weighted CTR
            daily_metrics['daily_ctr'] = (daily_metrics['daily_clicks'] / daily_metrics['daily_impressions'] * 100).fillna(0)

             # Weighted Position requires joining back impressions to original df or careful grouping
            query_df_merged = pd.merge(query_df, daily_metrics[['date', 'daily_impressions']], on='date', how='left')
            query_df_merged['position_numeric'] = pd.to_numeric(query_df_merged['position'], errors='coerce')
            query_df_merged['pos_x_imp'] = query_df_merged['position_numeric'] * query_df_merged['impressions']

            daily_pos_agg = query_df_merged.groupby('date').agg(
                 sum_pos_x_imp=('pos_x_imp', 'sum'),
                 sum_impressions_check=('impressions', 'sum') # Re-sum impressions to ensure alignment
            ).reset_index()

            daily_pos_agg['daily_position'] = (daily_pos_agg['sum_pos_x_imp'] / daily_pos_agg['sum_impressions_check']).fillna(query_df_merged['position_numeric'].mean()) # Fill NaN with overall mean if daily imp is 0

            # Merge all daily metrics
            daily_metrics = pd.merge(daily_metrics, daily_pos_agg[['date', 'daily_position']], on='date', how='left')

        else:
            daily_metrics = pd.DataFrame(columns=['date', 'daily_clicks', 'daily_impressions', 'daily_ctr', 'daily_position'])


        # Create tabs for charts
        tab1, tab2, tab3 = st.tabs(["Clicks & Impressions", "CTR Trend", "Position Trend"])

        with tab1:
            st.markdown("#### Daily Clicks and Impressions")
            if not daily_metrics.empty:
                fig = go.Figure()
                # Clicks trace (Left Y-axis)
                fig.add_trace(go.Scatter(
                    x=daily_metrics['date'],
                    y=daily_metrics['daily_clicks'],
                    mode='lines+markers',
                    name='Clicks',
                    line=dict(color='#3B82F6', width=2),
                    marker=dict(size=4)
                ))

                # Impressions trace (Right Y-axis)
                fig.add_trace(go.Scatter(
                    x=daily_metrics['date'],
                    y=daily_metrics['daily_impressions'],
                    mode='lines',
                    name='Impressions',
                    line=dict(color='#10B981', width=2, dash='dot'),
                    yaxis='y2' # Assign to secondary y-axis
                ))

                # Update layout with dual axes
                fig.update_layout(
                    # title="Clicks & Impressions Over Time", # Removed redundant title
                    xaxis_title="Date",
                    yaxis=dict(
                        title=dict(
                            text="Clicks",
                            font=dict(color='#3B82F6')
                        ),
                        tickfont=dict(color='#3B82F6'),
                        showgrid=False, # Cleaner look
                    ),
                    yaxis2=dict(
                        title=dict(
                            text="Impressions",
                            font=dict(color='#10B981')
                        ),
                        tickfont=dict(color='#10B981'),
                        anchor="x", # Anchored to the x-axis
                        overlaying="y", # Overlays the primary y-axis
                        side="right", # Position on the right
                        showgrid=False,
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02, # Position above chart
                        xanchor="right",
                        x=1
                    ),
                    height=450, # Adjusted height
                    hovermode="x unified", # Show tooltips for both traces on hover
                    margin=dict(l=50, r=50, t=50, b=50) # Adjust margins
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for Clicks & Impressions trend.")


        with tab2:
            st.markdown("#### Daily Average Click-Through Rate (CTR)")
            if not daily_metrics.empty:
                fig = px.line(
                    daily_metrics,
                    x='date',
                    y='daily_ctr',
                    # title="Click-Through Rate Over Time", # Redundant Title
                    labels={'date': 'Date', 'daily_ctr': 'CTR (%)'},
                    line_shape='spline' # Smoothed line
                )
                fig.update_traces(line=dict(color='#6366F1', width=3)) # Purple color
                fig.update_layout(
                    height=450,
                    hovermode="x unified",
                    yaxis_title="CTR (%)",
                    xaxis_title="Date"
                 )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for CTR trend.")

        with tab3:
            st.markdown("#### Daily Average Position")
            if not daily_metrics.empty:
                fig = px.line(
                    daily_metrics,
                    x='date',
                    y='daily_position',
                    # title="Average Position Over Time", # Redundant Title
                    labels={'date': 'Date', 'daily_position': 'Position'},
                    line_shape='spline' # Smoothed line
                )
                fig.update_traces(line=dict(color='#F59E0B', width=3)) # Orange color
                fig.update_layout(
                    height=450,
                    yaxis=dict(autorange="reversed"), # Reverse Y-axis (lower position is better)
                    hovermode="x unified",
                    yaxis_title="Average Position",
                    xaxis_title="Date"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for Position trend.")

        st.divider()

        # --- Top Queries and Pages ---
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('### Top Performing Queries')
            if not query_df.empty:
                top_queries = query_df.groupby('query').agg(
                    total_clicks=('clicks', 'sum'),
                    total_impressions=('impressions', 'sum'),
                    # Calculate overall CTR for the query
                    average_ctr=pd.NamedAgg(column='clicks', aggfunc=lambda x: (x.sum() / query_df.loc[x.index, 'impressions'].sum() * 100) if query_df.loc[x.index, 'impressions'].sum() > 0 else 0),
                     # Calculate weighted average position for the query
                    average_position=pd.NamedAgg(column='position_numeric', aggfunc=lambda x: (x * query_df.loc[x.index, 'impressions']).sum() / query_df.loc[x.index, 'impressions'].sum() if query_df.loc[x.index, 'impressions'].sum() > 0 else x.mean())
                ).sort_values('total_clicks', ascending=False).head(15).reset_index() # Show top 15
                top_queries = top_queries.round({'average_ctr': 2, 'average_position': 1}) # Round values

                fig = px.bar(
                    top_queries.sort_values('total_clicks', ascending=True), # Sort for horizontal bar chart
                    y='query', # Use query on Y for readability
                    x='total_clicks',
                    hover_data=['total_impressions', 'average_ctr', 'average_position'],
                    labels={'query': 'Query', 'total_clicks': 'Total Clicks'},
                    # title="Top 15 Queries by Clicks", # Redundant Title
                    color='total_clicks',
                    color_continuous_scale=px.colors.sequential.Blues,
                    orientation='h', # Horizontal bar chart
                    text='total_clicks' # Show clicks on bars
                )
                fig.update_layout(
                    height=600, # Taller for more queries
                    yaxis_title=None, # Remove Y axis title
                    xaxis_title="Total Clicks",
                    coloraxis_showscale=False, # Hide color scale bar
                    margin=dict(l=10, r=10, t=30, b=30)
                 )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("View Top Queries Data Table"):
                    st.dataframe(top_queries, use_container_width=True)
            else:
                 st.info("No query data available.")

        with col2:
            st.markdown('### Top Performing Pages')
            if not page_df.empty:
                 # Ensure position is numeric before aggregation
                 page_df['position_numeric'] = pd.to_numeric(page_df['position'], errors='coerce')

                 top_pages = page_df.groupby('page').agg(
                    total_clicks=('clicks', 'sum'),
                    total_impressions=('impressions', 'sum'),
                    average_ctr=pd.NamedAgg(column='clicks', aggfunc=lambda x: (x.sum() / page_df.loc[x.index, 'impressions'].sum() * 100) if page_df.loc[x.index, 'impressions'].sum() > 0 else 0),
                    average_position=pd.NamedAgg(column='position_numeric', aggfunc=lambda x: (x * page_df.loc[x.index, 'impressions']).sum() / page_df.loc[x.index, 'impressions'].sum() if page_df.loc[x.index, 'impressions'].sum() > 0 else x.mean())
                ).sort_values('total_clicks', ascending=False).head(15).reset_index() # Show top 15
                 top_pages = top_pages.round({'average_ctr': 2, 'average_position': 1}) # Round values

                # Clean page paths for display
                 def clean_page_path(path, site_url):
                     # Remove protocol and domain
                     path = path.replace('https://', '').replace('http://', '')
                     site_domain = site_url.replace('https://', '').replace('http://', '').split('/')[0]
                     if path.startswith(site_domain):
                         path = path.replace(site_domain, '', 1)
                     return path if path else '/' # Return root if empty

                 top_pages['page_path'] = top_pages['page'].apply(lambda x: clean_page_path(x, st.session_state.selected_site))


                 fig = px.bar(
                    top_pages.sort_values('total_clicks', ascending=True), # Sort for horizontal bar chart
                    y='page_path', # Use page path on Y
                    x='total_clicks',
                    hover_data=['total_impressions', 'average_ctr', 'average_position', 'page'], # Show full URL on hover
                    labels={'page_path': 'Page Path', 'total_clicks': 'Total Clicks'},
                    # title="Top 15 Pages by Clicks", # Redundant Title
                    color='total_clicks',
                    color_continuous_scale=px.colors.sequential.Greens,
                    orientation='h', # Horizontal bar chart
                    text='total_clicks' # Show clicks on bars
                )
                 fig.update_layout(
                    height=600, # Taller for more pages
                    yaxis_title=None, # Remove Y axis title
                    xaxis_title="Total Clicks",
                    coloraxis_showscale=False, # Hide color scale bar
                    margin=dict(l=10, r=10, t=30, b=30) # Adjust left margin if paths are long
                 )
                 fig.update_traces(textposition='outside')
                 st.plotly_chart(fig, use_container_width=True)
                 with st.expander("View Top Pages Data Table"):
                     # Show relevant columns
                     st.dataframe(top_pages[['page_path', 'total_clicks', 'total_impressions', 'average_ctr', 'average_position', 'page']], use_container_width=True)
            else:
                st.info("No page data available.")

        st.divider()

        # --- Gemini Analysis ---
        if st.session_state.gemini_initialized:
            st.markdown('<h2 class="sub-header">🤖 Gemini AI Analysis & Recommendations</h2>', unsafe_allow_html=True)

            if st.session_state.performance_summary or st.session_state.trend_analysis:
                 # Display Key Insights first
                 if st.session_state.performance_summary:
                    st.markdown("### Key Performance Insights")
                    summary_lines = st.session_state.performance_summary.strip().split('\n')
                    for line in summary_lines:
                        clean_line = line.strip().lstrip('*- ') # Remove bullets/leading spaces
                        if clean_line:
                            st.markdown(f'<div class="insight-card">{clean_line}</div>', unsafe_allow_html=True)
                    st.markdown("---") # Separator

                 # Display Full Analysis in an expander
                 if st.session_state.trend_analysis:
                    st.markdown("### Detailed Analysis")
                    with st.expander("View Full Gemini Analysis", expanded=False):
                        st.markdown(st.session_state.trend_analysis)
                    st.markdown("---") # Separator


                 # Recommendation section (Optional - generate on demand?)
                 st.markdown('### Action Recommendations')
                 st.info("Recommendations are based on the patterns identified in your data.")

                 # Generate specific recommendations only if analysis exists
                 if st.session_state.gemini_model and st.session_state.trend_analysis:
                    # Button to generate recommendations if needed, or display if already generated
                    if 'recommendations' not in st.session_state:
                         st.session_state.recommendations = None

                    if st.button("Generate Recommendations", key="gen_reco"):
                         recommendation_prompt = f"""
                        Based on the previous SEO analysis provided for {st.session_state.selected_site}, generate 3-5 specific, actionable recommendations to improve SEO performance.

                        Focus on:
                        - Capitalizing on top performers.
                        - Improving pages/queries with high impressions but low CTR.
                        - Addressing potential issues highlighted in the analysis.

                        Format your response as a numbered list. Each recommendation should start with an action verb and be 1-2 sentences max.
                        Example: 1. Optimize the title tag and meta description for the page '/example-page' to improve its low CTR.
                        """
                         with st.spinner("Generating recommendations..."):
                            try:
                                recommendations_response = st.session_state.gemini_model.generate_content(recommendation_prompt)
                                st.session_state.recommendations = recommendations_response.text
                                st.markdown(st.session_state.recommendations)
                            except Exception as e:
                                st.error(f"Failed to generate recommendations: {str(e)}")
                    elif st.session_state.recommendations:
                         # Display previously generated recommendations
                         st.markdown(st.session_state.recommendations)

            else:
                 st.info("Click 'Fetch Data & Analyze' to generate AI insights.")
        else:
            st.info("Initialize Gemini API in the sidebar to enable AI analysis.")

# Optional: Add a footer
st.markdown("---")
st.caption("SEO Performance Analyzer v1.0")

# --- End of Script ---
# No need for if __name__ == "__main__": in Streamlit apps typically
