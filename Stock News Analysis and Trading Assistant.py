# ⬅️ Your existing imports
import streamlit as st
import requests
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import yfinance as yf

# YouTube API
YOUTUBE_API_KEY = "your-key"
def search_youtube_videos(query, max_results=5):
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=video&maxResults={max_results}&key={YOUTUBE_API_KEY}"
    response = requests.get(url).json()
    videos = []
    if "items" in response:
        for item in response["items"]:
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            videos.append((title, video_id))
    return videos

# LSTM Model
class LSTMSentiment(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMSentiment, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        output = self.fc(self.dropout(lstm_out[:, -1, :]))
        return output

# Load model
model = LSTMSentiment(input_dim=5000, hidden_dim=256, output_dim=3, num_layers=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("lstm_sentiment_model.pth", map_location=device))
model.to(device)
model.eval()

# Data & vectorizer
train_df = pd.read_csv("sent_train.csv")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(train_df["text"]).toarray()
y = train_df["label"].values

def predict_sentiment(text):
    tokens = vectorizer.transform([text]).toarray()
    input_tensor = torch.tensor(tokens, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    sentiment = torch.argmax(output, dim=1).item()
    return "Positive" if sentiment == 2 else "Neutral" if sentiment == 1 else "Negative"

def get_finviz_news(ticker):
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None
    soup = BeautifulSoup(response.text, 'html.parser')
    news_table = soup.find(id='news-table')
    parsed_news = []
    if news_table:
        for row in news_table.findAll('tr'):
            title = row.a.text if row.a else "No Title"
            parsed_news.append(title)
    return parsed_news

def highlight_sentiment(val):
    color = "#90EE90" if val == "Positive" else "#FFD700" if val == "Neutral" else "#FF6347"
    return f'background-color: {color}; color: black; font-weight: bold;'

# Simple Technical Analysis
def get_technical_signal(ticker):
    try:
        data = yf.download(ticker, period="3mo", interval="1d")
        data["SMA7"] = data["Close"].rolling(window=7).mean()
        data["SMA14"] = data["Close"].rolling(window=14).mean()
        if data["SMA7"].iloc[-1] > data["SMA14"].iloc[-1]:
            return "Buy"
        elif data["SMA7"].iloc[-1] < data["SMA14"].iloc[-1]:
            return "Sell"
        else:
            return "Hold"
    except:
        return "Data Error"

def final_decision(sentiment, tech_signal, risk_level):
    if sentiment == "Positive" and tech_signal == "Buy":
        return "Strong Buy" if risk_level != "Low" else "Cautious Buy"
    elif sentiment == "Negative" and tech_signal == "Sell":
        return "Sell"
    else:
        return "Hold"

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Sentiment Analysis", "Trading Assistant", "Model Evaluation", "About"])

# Home
if page == "Home":
    st.title("Welcome to Stock News Sentiment Analysis and Trading Assistant")
    st.write("""
    ### 📈 Real-Time Stock Sentiment Analysis
    Stay ahead in the stock market with real-time news sentiment insights. Our intelligent assistant analyzes financial headlines using natural language processing to detect market sentimet and provide various kinds of analysis, helping you make smarter, data-driven trading decisions. Whether you're a beginner or an experienced trader, our tool empowers you with timely information and intuitive visualizations to navigate market trends confidently.
             """)

# Sentiment Analysis
elif page == "Sentiment Analysis":
    st.title("📊 Real-Time Stock News Sentiment Analyzer")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")
    if st.button("Fetch News & Analyze"):
        news_headlines = get_finviz_news(ticker.upper())
        if news_headlines:
            st.subheader(f"📰 Latest News for {ticker.upper()}")
            results = []
            for news in news_headlines[:50]:
                sentiment = predict_sentiment(news)
                results.append([news, sentiment])
            df = pd.DataFrame(results, columns=["News Headline", "Predicted Sentiment"])
            st.dataframe(df.style.applymap(highlight_sentiment, subset=["Predicted Sentiment"]))

        else:
            st.error("⚠️ Failed to fetch news.")

    if st.button("Search Related Videos"):
        results = search_youtube_videos(ticker + " stock analysis")
        for title, video_id in results:
            st.subheader(title)
            st.markdown(f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)

# Trading Assistant
elif page == "Trading Assistant":
    st.title("🤖 Personalized Trading Assistant")
    st.markdown("""
    <style>
    /* 🌐 GLOBAL APP STYLING */
    html, body, [class*="css"] {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', sans-serif;
        color: #2e3a59;
    }

    /* 🎯 TITLE STYLE */
    h1 {
        font-size: 2.5rem;
        text-align: center;
        margin-top: 1rem;
        color: #00000;
        font-weight: 800;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }

    /* ⬛ CONTAINER CARD */
    .element-container:has(.stTabs) {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.07);
    }

    /* 📥 TEXT INPUT + SELECTBOX */
    .stTextInput > div > div > input, .stSelectbox div[data-baseweb] {
        background-color: #ffffff !important;
        padding: 12px !important;
        border-radius: 10px !important;
        border: 1px solid #d4dce2 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        font-size: 1rem;
    }

    /* 🔘 BUTTON STYLE */
    button[kind="primary"] {
        background: linear-gradient(to right, #1d8cf8, #3358f4) !important;
        color: #fff !important;
        font-size: 1rem;
        font-weight: bold;
        padding: 0.6rem 1.3rem;
        border: none;
        border-radius: 8px !important;
        margin-top: 1.2rem;
        transition: all 0.3s ease;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(to right, #3358f4, #1d8cf8) !important;
        transform: scale(1.03);
    }

    /* 📑 TABS DESIGN */
    .stTabs [role="tab"] {
        font-size: 1rem;
        font-weight: 600;
        padding: 10px 18px;
        border-radius: 10px 10px 0 0;
        background-color: #eaf1f8;
        color: #2e3a59;
        margin-right: 8px;
        transition: background-color 0.3s;
    }
    .stTabs [role="tab"]:hover {
        background-color: #d8e7f3;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #ffffff;
        color: #1a73e8;
        border-bottom: 3px solid #1a73e8;
    }

    /* 📊 CHARTS + FIGURE BLOCKS */
    .stPlotlyChart, .stPyplotChart, .stAltairChart, .element-container:has(canvas) {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.06);
    }

    /* 📌 HEADLINES SENTIMENT */
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        font-weight: 700;
        color: #2e3a59;
    }

    /* 🧾 TEXT CONTENT */
    .stMarkdown {
        font-size: 1.02rem;
        color: #37474f;
    }

    /* ✅ SUBHEADINGS */
    .stSubheader {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2e3a59;
        margin-top: 1rem;
    }

    /* ⚠️ WARNINGS & ERRORS */
    .stAlert {
        border-radius: 10px;
        padding: 1rem;
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    /* 📈 LINE CHART IN HISTORY TAB */
    .stLineChart {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
    }

    /* 🧮 METRIC BOXES IN FUNDAMENTALS */
    .stMarkdown p {
        background-color: #f0f4fa;
        padding: 0.7rem 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }

    /* 🗃️ FINANCIAL DATA NUMBERS */
    .stMarkdown strong {
        color: #1a237e;
        font-weight: 600;
    }

    /* 🎯 CHARTS HEIGHT FIX (OPTIONAL) */
    .stPlotlyChart > div, .stPyplotChart > div {
        height: 400px !important;
    }
    </style>
""", unsafe_allow_html=True)


    risk_tolerance = st.selectbox("Risk Tolerance Level", ["Low", "Medium", "High"])

    st.markdown("""<style>
        input[type="text"] {
            color: black !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Text input
    ticker = st.text_input("Enter Stock Ticker for Analysis", "AAPL")

    if st.button("Analyze"):
        tabs = st.tabs(["Fundamentals", "Technical Chart", "Sentiment", "Buy/Sell Signal", "Stock History"])

        # 1. Fundamentals
        with tabs[0]:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                pe = info.get("trailingPE", "N/A")
                eps = info.get("trailingEps", "N/A")
                mcap = info.get('marketCap', 0)
                st.write(f"**P/E Ratio:** {pe}")
                st.write(f"**EPS:** {eps}")
                st.write(f"**Market Cap:** {mcap}")
                
            except:
                st.error("Could not fetch fundamental data.")

        # 2. Technical Chart
        with tabs[1]:
            try:
                df = yf.download(ticker, period="3mo", interval="1d")
                df["SMA7"] = df["Close"].rolling(window=7).mean()
                df["SMA14"] = df["Close"].rolling(window=14).mean()
                fig, ax = plt.subplots()
                ax.plot(df.index, df["Close"], label="Close Price")
                ax.plot(df.index, df["SMA7"], label="SMA7")
                ax.plot(df.index, df["SMA14"], label="SMA14")
                ax.set_title("Technical Chart with SMA")
                ax.legend()
                st.pyplot(fig)
            except:
                st.error("Failed to render chart")

        # 3. Sentiment
        with tabs[2]:
            headlines = get_finviz_news(ticker.upper())
            if headlines:
                sentiments = [predict_sentiment(h) for h in headlines[:50]]
                pos = sentiments.count("Positive")
                neg = sentiments.count("Negative")
                neu = sentiments.count("Neutral")
                st.write(f"**Summary:** {pos} Positive | {neu} Neutral | {neg} Negative")
                plt.figure(figsize=(5,3))
                sns.barplot(x=["Positive", "Neutral", "Negative"], y=[pos, neu, neg])
                st.pyplot(plt)
            else:
                st.warning("No headlines found.")

        # 4. Signal
        with tabs[3]:
            sentiment = predict_sentiment(" ".join(get_finviz_news(ticker.upper())[:3]))
            tech_signal = get_technical_signal(ticker)
            decision = final_decision(sentiment, tech_signal, risk_tolerance)
            st.write(f"**Buy/Sell Signal:** {decision}")


        # 5. Stock History
        with tabs[4]:
            ALPHA_VANTAGE_API_KEY = "your-key"  

            def get_fundamentals_av(ticker):
                url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
                response = requests.get(url)
                return response.json()

            def get_current_price_av(ticker):
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
                response = requests.get(url)
                return response.json().get("Global Quote", {})

            try:
                fundamentals = get_fundamentals_av(ticker)
                quote = get_current_price_av(ticker)

                pe_ratio = fundamentals.get("PERatio", "N/A")
                eps = fundamentals.get("EPS", "N/A")
                market_cap = fundamentals.get("MarketCapitalization", "N/A")
                current_price = quote.get("05. price", "N/A")

                st.write(f"**Current Price:** ${current_price}")
                st.write(f"**P/E Ratio:** {pe_ratio}")
                st.write(f"**EPS:** {eps}")
                st.write(f"**Market Cap:** {int(market_cap):,}" if market_cap != "N/A" else "**Market Cap:** N/A")
            except Exception as e:
                st.error("Failed to fetch data from Alpha Vantage. Ensure the ticker is valid and API key is active.")
            import datetime
            import matplotlib.pyplot as plt

            
            def fetch_stock_history(ticker):
                url = (
                    f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
                    f"&symbol={ticker}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}"
                )
                response = requests.get(url)
                data = response.json()

                if "Time Series (Daily)" not in data:
                    st.write("API raw response:", data)
                    return None

                time_series = data["Time Series (Daily)"]
                dates = []
                close_prices = []

                for date, values in sorted(time_series.items(), reverse=False):
                    dates.append(datetime.datetime.strptime(date, "%Y-%m-%d"))
                    close_prices.append(float(values["4. close"]))

                return pd.DataFrame({"Date": dates, "Close": close_prices})

        
            st.subheader("📈 Stock Price Chart (Last 100 Days)")
            history_df = fetch_stock_history(ticker.upper())
            if history_df is not None:
                st.line_chart(history_df.set_index("Date")["Close"])
            else:          
                st.error("📉 Couldn't load historical chart data. API error or invalid ticker.")



            def prepare_growth_and_chart_data(time_series):
                df = pd.DataFrame.from_dict(time_series, orient="index", dtype=float)
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                df.rename(columns={"5. adjusted close": "Adj Close"}, inplace=True)
                return df

elif page == "Model Evaluation":
    # Evaluate model
            st.subheader("📊 LSTM Sentiment Model Evaluation")
            with torch.no_grad():
                input_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                preds = model(input_tensor).argmax(dim=1).cpu().numpy()

            acc = accuracy_score(y, preds)
            report = classification_report(y, preds, output_dict=True)
            conf = confusion_matrix(y, preds)
            st.write(f"**Accuracy:** {acc*100:.2f}%")
            st.dataframe(pd.DataFrame(report).transpose())
            st.write("**Confusion Matrix:**")
            fig, ax = plt.subplots()
            sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg","Neu","Pos"], yticklabels=["Neg","Neu","Pos"])
            st.pyplot(fig)


# About
elif page == "About":
    st.title("📌 About this App")
    st.write("""
    This app uses a PyTorch-based LSTM model for analyzing market sentiment from news headlines and provides a trading assistant interface with technical and fundamental analysis.
    """)
