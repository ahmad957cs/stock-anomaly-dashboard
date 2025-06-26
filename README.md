# 📈 Stock Anomaly Detection Dashboard

A Streamlit web app to **detect anomalies** in historical stock price data using both statistical models and machine learning techniques.

---

## 🎯 Objective

Build a tool that:

- Fetches real-time stock data using **Yahoo Finance**
- Calculates key **financial indicators** (SMA, EMA, RSI, Bollinger Bands)
- Detects unusual price patterns using:
  - 🧠 **Isolation Forest** (unsupervised anomaly detection)
  - 🔮 **Prophet** (forecast-based anomaly detection)
- Visualizes anomalies with **interactive plots**
- Enables **export** of cleaned and analyzed data

---

## 💡 Features

| Feature                        | Description                                                   |
|-------------------------------|---------------------------------------------------------------|
| 📊 Interactive Dashboard       | Choose your own stock ticker (AAPL, TSLA, etc.)               |
| 🔁 Time Range Selection        | Define custom start and end dates                             |
| 📐 Technical Indicators        | SMA, EMA, RSI, Bollinger Bands using `ta` library             |
| 🧠 Machine Learning Detection  | Isolation Forest to detect anomalies based on market behavior |
| 🔮 Time-Series Forecasting     | Forecast future trends using Facebook Prophet                 |
| 📌 Anomaly Highlighting        | Red/green points to highlight spikes and drops                |
| ⬇ CSV Export                  | Download cleaned data and anomalies for reporting             |

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [yFinance](https://pypi.org/project/yfinance/)
- [TA](https://technical-analysis-library-in-python.readthedocs.io/en/latest/)
- [Prophet (Meta)](https://facebook.github.io/prophet/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)

---

## 🚀 Launch the App

**Live Demo**:  
👉 [https://your-username.streamlit.app](https://your-username.streamlit.app)  
_(Replace with your actual link after deployment)_

---

## 📦 Installation (For Local Use)

```bash
git clone https://github.com/your-username/stock-anomaly-dashboard.git
cd stock-anomaly-dashboard
pip install -r requirements.txt
streamlit run stock_dashboard.py

File Structure

├── stock_dashboard.py        # Main Streamlit app
├── requirements.txt          # Project dependencies
└── README.md                 # You're here
## 👤 Author
Ahmad Gul
BS Computer Science (AI Specialization)
LinkedIn Profile (www.linkedin.com/in/ahmad-gul-8365b0307)

