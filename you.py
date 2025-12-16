# ⬅️ Your existing imports
import streamlit as st
import requests
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer
import yfinance as yf

# YouTube API
YOUTUBE_API_KEY = "AIzaSyCVAXk2f61gJvrZ64Ntf_xN8xLzUSjqwsU"
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

df = pd.read_csv("sent_train.csv")
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(df["text"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMSentiment(input_dim=5000, hidden_dim=256, output_dim=3, num_layers=2).to(device)
model.load_state_dict(torch.load("lstm_sentiment_model.pth", map_location=device))
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
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

# NEW: Simple Technical Analysis (SMA crossover)
def get_technical_signal(ticker):
    try:
        data = yf.download(ticker, period="1mo", interval="1d")
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

# NEW: Decision-making logic
def final_decision(sentiment, tech_signal, risk_level):
    if sentiment == "Positive" and tech_signal == "Buy":
        return "Strong Buy" if risk_level != "Low" else "Cautious Buy"
    elif sentiment == "Negative" and tech_signal == "Sell":
        return "Sell"
    else:
        return "Hold"

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Sentiment Analysis", "Trading Assistant", "About"])

# Home
if page == "Home":
    st.title("Welcome to Stock Sentiment Analyzer")
    st.write("""
    ### 📈 Real-Time Stock Sentiment Analysis
    This application helps investors analyze real-time stock news sentiment using a deep learning model.
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

    st.header("🔍 Enter Your Trading Preferences")

    trading_type = st.selectbox("Select Trading Type", ["Intraday", "Forex"])
    sectors = st.text_input("Preferred Sectors (comma-separated)", "Technology, Finance")
    currency_pairs = st.text_input("Preferred Currency Pairs (if Forex)", "USD/INR, EUR/USD")
    risk_tolerance = st.selectbox("Risk Tolerance Level", ["Low", "Medium", "High"])
    trading_capital = st.number_input("Trading Capital (optional)", min_value=0.0, format="%.2f")

    ticker = st.text_input("Enter Stock Ticker for Analysis", "AAPL")
    if st.button("Analyze"):
        st.subheader("📈 Real-Time Sentiment Analysis")
        headlines = get_finviz_news(ticker.upper())
        sentiment_results = []
        if headlines:
            for news in headlines[:50]:
                sentiment = predict_sentiment(news)
                sentiment_results.append(sentiment)
            sentiment_df = pd.DataFrame({
                "Headlines": headlines[:50],
                "Sentiment": sentiment_results
            })
            st.dataframe(sentiment_df.style.applymap(highlight_sentiment, subset=["Sentiment"]))

            # Sentiment Summary
            pos = sentiment_results.count("Positive")
            neg = sentiment_results.count("Negative")
            neu = sentiment_results.count("Neutral")
            st.markdown(f"**Summary:** {pos} Positive | {neu} Neutral | {neg} Negative")

        else:
            st.warning("No headlines found for this ticker.")

        st.subheader("📊 Fundamental Analysis")
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info
            pe_ratio = info.get("trailingPE", "N/A")
            eps = info.get("trailingEps", "N/A")
            market_cap = info.get("marketCap", "N/A")
            st.write(f"**P/E Ratio:** {pe_ratio}")
            st.write(f"**EPS:** {eps}")
            st.write(f"**Market Cap:** {market_cap:,}" if market_cap != "N/A" else "**Market Cap:** N/A")
        except Exception as e:
            st.error("Failed to fetch fundamental data. Ensure ticker is valid.")

        st.subheader("📉 Risk & Reward Assessment")

        score = (pos - neg) + (1 if risk_tolerance == "High" else 0)
        if score > 2:
            st.success("📈 Recommendation: Good opportunity. Consider a BUY signal based on sentiment.")
        elif score <= 0 and risk_tolerance == "Low":
            st.error("⚠️ Recommendation: Risky. Consider AVOIDING this stock.")
        else:
            st.warning("🤔 Recommendation: Mixed signals. Wait for better confirmation.")


# About
elif page == "About":
    st.title("📌 About this App")
    st.write("""
    This app uses a PyTorch-based LSTM model for analyzing market sentiment from news headlines and provides a trading assistant interface with basic technical indicators.
    """)
