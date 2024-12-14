import os
import logging
import datetime
import warnings
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import nltk
from io import StringIO
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv  # Added for .env support
import openai  # Standard OpenAI import
from sklearn.linear_model import LinearRegression
from st_aggrid import AgGrid, GridOptionsBuilder  # For interactive data table
from bs4 import BeautifulSoup  # Moved import to top for consistency

# =====================================
# .env Instructions:
# Ensure your .env file has exactly two lines:
# OPENAI_API_KEY=sk-xxxx...
# ELEVENLABS_API_KEY=xxxxxxx
#
# No extra lines, code, or comments.
#
# Make sure these keys are valid. If OpenAI returns 401, your OPENAI_API_KEY is invalid.
# =====================================

# Load environment variables from .env file
load_dotenv()

# =====================================
# Initial Setup and Configuration
# =====================================

st.set_page_config(page_title="Engagement & Sentiment Analysis", layout="wide")
logging.basicConfig(level=logging.INFO)

# Suppress specific openpyxl warning about default styles
warnings.filterwarnings("ignore", category=UserWarning, message="Workbook contains no default style")

# Ensure NLTK VADER lexicon is available only once
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# API Keys (ensure these are set as environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# VOICE_ID remains hardcoded as per user's request
VOICE_ID = "RyD9qVuVMj2Yiq1KOQ2A"  # Specified voice ID

# Validate API Keys
openai_available = OPENAI_API_KEY not in (None, "", "your_openai_api_key_here")
elevenlabs_available = ELEVENLABS_API_KEY not in (None, "", "your_elevenlabs_api_key_here")

if not openai_available:
    st.warning("OpenAI API key not set or invalid. No chat insights available.")
if not elevenlabs_available:
    st.info("ElevenLabs API key not set. Audio summaries won't be available.")

# Set OpenAI API Key
if openai_available:
    openai.api_key = OPENAI_API_KEY

# ElevenLabs configuration
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
HEADERS = {
    "xi-api-key": ELEVENLABS_API_KEY,
    "Content-Type": "application/json"
}

# Color map for sentiment categories
COLOR_MAP = {"Negative": "red", "Neutral": "gray", "Positive": "blue"}

# =====================================
# Helper & Utility Functions
# =====================================

def categorize_sentiment(score: float) -> str:
    """
    Convert a VADER compound sentiment score into categories.
    compound >= 0.05 => Positive
    compound <= -0.05 => Negative
    otherwise => Neutral
    """
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"

@st.cache_data
def compute_sentiments(texts):
    """
    Compute sentiment categories from extracted text using VADER, cached for performance.
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for text in texts:
        scores = analyzer.polarity_scores(text)
        sentiment = categorize_sentiment(scores["compound"])
        sentiments.append(sentiment)
    return sentiments

def generate_audio_summary(text: str):
    """
    Generate an audio summary using ElevenLabs if the API key and VOICE_ID are provided.
    """
    if not (elevenlabs_available and VOICE_ID):
        if not elevenlabs_available:
            st.info("ElevenLabs API Key not set. Skipping audio summary.")
        if not VOICE_ID:
            st.warning("VOICE_ID not set. Please specify a valid VOICE_ID.")
        return
    try:
        body = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        response = requests.post(f"{ELEVENLABS_URL}/{VOICE_ID}", headers=HEADERS, json=body)
        if response.status_code == 200:
            with open("audio_summary.mp3", "wb") as f:
                f.write(response.content)
            st.audio("audio_summary.mp3")
        else:
            st.error("ElevenLabs API error. Check your ELEVENLABS_API_KEY and VOICE_ID.")
    except Exception as e:
        st.error(f"Audio summary error: {e}")

def query_openai(prompt: str) -> str:
    """
    Query the OpenAI ChatCompletion API for analysis or recommendations.
    """
    if not openai_available:
        return "OpenAI API key not set."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except openai.error.AuthenticationError:
        st.error("OpenAI returned 401 Unauthorized. Check your OPENAI_API_KEY.")
        return "Authentication Error: Invalid OpenAI API key."
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return "Error querying OpenAI."

def process_html_files(html_files):
    """
    Extract text from uploaded HTML files and compute sentiments using VADER.
    """
    texts = []
    for f in html_files:
        try:
            content = f.getvalue().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            texts.append(text)
        except Exception as e:
            st.error(f"Error reading HTML file {f.name}: {e}")
    return texts

def add_forecast(df: pd.DataFrame):
    """
    Add a linear regression forecast of future impressions based on the date.
    """
    if "Post publish date" in df.columns and "Impressions" in df.columns:
        df_f = df.dropna(subset=["Impressions", "Post publish date"])
        if len(df_f) > 5:
            df_f["date_ordinal"] = df_f["Post publish date"].map(datetime.datetime.toordinal)
            X = df_f[["date_ordinal"]]
            y = df_f["Impressions"]
            model = LinearRegression().fit(X, y)
            future_date = df_f["Post publish date"].max() + pd.Timedelta(days=30)
            future_ordinal_df = pd.DataFrame({"date_ordinal": [future_date.toordinal()]})
            future_pred = model.predict(future_ordinal_df)[0]
            st.write(f"**Impressions Forecast (30 days):** {future_pred:.2f}")

def display_visuals(df: pd.DataFrame):
    """
    Create and display visualizations for engagement data with sentiment.
    """
    st.write("#### Visualizations")
    show_pie = st.checkbox("Show Sentiment Distribution Pie", value=True)
    show_scatter = st.checkbox("Show Engagement Rate vs Impressions Scatter", value=True)
    show_hist = st.checkbox("Show Distribution of Engagement Rate Histogram", value=True)
    show_bar = st.checkbox("Show Top Posts by Engagement Rate Bar", value=True)
    show_line = st.checkbox("Show Impressions Over Time Line", value=True)

    if show_pie and "Sentiment" in df.columns:
        sentiment_counts = df["Sentiment"].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map=COLOR_MAP
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    if show_scatter and "Engagement Rate (%)" in df.columns and "Impressions" in df.columns:
        scatter_fig = px.scatter(
            df,
            x="Impressions",
            y="Engagement Rate (%)",
            color="Sentiment" if "Sentiment" in df.columns else None,
            template="plotly_dark",
            title="Engagement Rate vs Impressions",
            hover_data=["Post URL", "Engagements"]
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    if show_hist and "Engagement Rate (%)" in df.columns:
        hist_fig = px.histogram(
            df,
            x="Engagement Rate (%)",
            nbins=30,
            title="Distribution of Engagement Rate",
            template="plotly_dark"
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    if show_bar and "Engagement Rate (%)" in df.columns and "Post URL" in df.columns:
        sorted_data = df.sort_values("Engagement Rate (%)", ascending=False).head(10)
        bar_fig = px.bar(
            sorted_data,
            x="Engagement Rate (%)",
            y="Post URL",
            orientation="h",
            title="Top 10 Posts by Engagement Rate",
            template="plotly_dark",
            color="Sentiment",
            color_discrete_map=COLOR_MAP
        )
        bar_fig.update_yaxes(autorange="reversed")
        st.plotly_chart(bar_fig, use_container_width=True)

    if show_line and "Post publish date" in df.columns and "Impressions" in df.columns:
        time_data = df.dropna(subset=["Post publish date"]).sort_values("Post publish date")

        if not time_data.empty:
            # Convert Pandas Timestamps to Python datetime.datetime objects
            min_date = time_data["Post publish date"].min().to_pydatetime()
            max_date = time_data["Post publish date"].max().to_pydatetime()
            date_range = st.slider(
                "Select Date Range for Impressions Over Time",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD"
            )
            line_filtered = time_data[
                (time_data["Post publish date"] >= date_range[0]) &
                (time_data["Post publish date"] <= date_range[1])
            ]

            line_fig = px.line(
                line_filtered,
                x="Post publish date",
                y="Impressions",
                title="Impressions Over Time",
                template="plotly_dark"
            )
            st.plotly_chart(line_fig, use_container_width=True)

    # Interactive Data Table using AgGrid
    st.write("### Interactive Engagement Data Table")
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection('single', use_checkbox=True)
    grid_options = gb.build()

    AgGrid(
        df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode='MODEL_CHANGED',
        theme='streamlit'  # Changed from 'dark' to a valid theme
    )

# =====================================
# Main Flow
# =====================================

st.title("Engagement & Sentiment Analysis")

st.write("""
Use the sidebar to upload files, apply filters, and analyze data. After analysis, you can request OpenAI insights or listen to an audio summary (if keys are provided).
""")

# File Uploads
uploaded_html_files = st.sidebar.file_uploader(
    "Upload HTML files (Optional)", type=["html"], accept_multiple_files=True
)
uploaded_excel_file = st.sidebar.file_uploader(
    "Upload Engagement Excel file (Required)", type=["xlsx"]
)

# Filters
start_date = st.sidebar.date_input(
    "Start Date", value=None, help="Select the start date for filtering posts."
)
end_date = st.sidebar.date_input(
    "End Date", value=None, help="Select the end date for filtering posts."
)
if start_date and end_date and start_date > end_date:
    st.sidebar.error("Start date must be before end date.")

sentiment_filter_options = ["Positive", "Neutral", "Negative"]
sentiment_selected = st.sidebar.multiselect(
    "Filter by Sentiment", sentiment_filter_options,
    default=sentiment_filter_options,
    help="Select which sentiments to include in the filtered data."
)
er_threshold = st.sidebar.slider(
    "Minimum Engagement Rate (%)", 0.0, 100.0, 0.0, step=0.1,
    help="Filter out posts with an engagement rate below this threshold."
)

# OpenAI Query
prompt = st.sidebar.text_area(
    "Ask OpenAI a question or analysis request:",
    help="Enter a prompt to ask OpenAI for insights after analyzing data."
)
analyze_button = st.sidebar.button(
    "Analyze Data", help="Click to analyze the uploaded Excel data and apply filters."
)
insights_button = st.sidebar.button(
    "Get OpenAI Insights", help="Click to request insights from OpenAI after data is analyzed."
)

# Analyze Button Logic
if analyze_button:
    if not uploaded_excel_file:
        st.error("Please upload an Excel file for engagement analysis.")
    else:
        try:
            df = pd.read_excel(uploaded_excel_file, sheet_name="TOP POSTS", header=2)
        except Exception as e:
            st.error(f"Error reading Excel: {e}")
            df = None

        if df is not None:
            required_cols = {"Post URL", "Post publish date", "Engagements", "Impressions"}
            missing = required_cols - set(df.columns)
            if missing:
                st.error(f"Missing required columns: {missing}")
                st.session_state["filtered_data"] = None
            else:
                df = df[list(required_cols)].copy()
                df["Engagements"] = pd.to_numeric(df["Engagements"], errors="coerce").fillna(0)
                df["Impressions"] = pd.to_numeric(df["Impressions"], errors="coerce").fillna(0)
                df["Engagement Rate (%)"] = np.where(
                    df["Impressions"] > 0,
                    (df["Engagements"] / df["Impressions"]) * 100,
                    np.nan
                )
                df.dropna(subset=["Post URL", "Post publish date"], inplace=True)
                df["Post publish date"] = pd.to_datetime(df["Post publish date"], errors="coerce")
                df.reset_index(drop=True, inplace=True)

                # HTML sentiments
                if uploaded_html_files:
                    texts = process_html_files(uploaded_html_files)
                    sentiments = compute_sentiments(texts) if texts else []
                    df["Sentiment"] = "Neutral"
                    length = min(len(df), len(sentiments))
                    if length > 0:
                        df.loc[:length-1, "Sentiment"] = sentiments[:length]
                else:
                    df["Sentiment"] = "Neutral"

                # Apply filters
                filtered_data = df.copy()
                if start_date and end_date:
                    filtered_data = filtered_data[
                        (filtered_data["Post publish date"] >= pd.to_datetime(start_date)) &
                        (filtered_data["Post publish date"] <= pd.to_datetime(end_date))
                    ]
                if sentiment_selected:
                    filtered_data = filtered_data[
                        filtered_data["Sentiment"].isin(sentiment_selected)
                    ]
                if er_threshold > 0.0:
                    filtered_data = filtered_data[
                        filtered_data["Engagement Rate (%)"] >= er_threshold
                    ]

                if filtered_data.empty:
                    st.warning("No data available after applying filters.")
                    # Offer a reset button
                    if st.button("Reset Filters"):
                        st.experimental_rerun()
                    st.session_state["filtered_data"] = None
                else:
                    # Make Post URL clickable links
                    filtered_data["Post URL"] = filtered_data["Post URL"].apply(lambda x: f"[{x}]({x})")

                    st.write("### Filtered Engagement Data")
                    # Use st_aggrid for interactive table
                    gb = GridOptionsBuilder.from_dataframe(filtered_data)
                    gb.configure_pagination()
                    gb.configure_side_bar()
                    gb.configure_selection('single', use_checkbox=True)
                    grid_options = gb.build()
                    AgGrid(
                        filtered_data,
                        gridOptions=grid_options,
                        height=300,
                        theme='streamlit'  # Changed from 'dark' to a valid theme
                    )

                    display_visuals(filtered_data)
                    add_forecast(filtered_data)

                    if "Engagement Rate (%)" in filtered_data.columns:
                        avg_er = filtered_data["Engagement Rate (%)"].mean()
                    else:
                        avg_er = 0.0
                    sentiments_counts = dict(filtered_data["Sentiment"].value_counts()) if "Sentiment" in filtered_data.columns else {}
                    summary_text = f"Average Engagement Rate: {avg_er:.2f}%\nSentiment Counts: {sentiments_counts}"

                    if elevenlabs_available:
                        generate_audio_summary(summary_text)

                    st.session_state["filtered_data"] = filtered_data
                    st.session_state["summary_text"] = summary_text
        else:
            st.error("No valid data to analyze.")
            st.session_state["filtered_data"] = None

# Insights Button Logic
if insights_button:
    # Only proceed if data has been analyzed
    if "filtered_data" not in st.session_state or st.session_state["filtered_data"] is None:
        st.info("No analyzed data available. Please click 'Analyze Data' first.")
    else:
        # Check if prompt is provided
        if prompt.strip():
            st.write("### OpenAI Insights")
            insight = query_openai(prompt)
            st.write(insight)
        else:
            st.info("Please enter a prompt before requesting OpenAI insights.")
