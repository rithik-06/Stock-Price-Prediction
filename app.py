
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import numpy as np

# Page config
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(to bottom, #0f0c29, #302b63, #24243e);
    }
    h1 {
        color: #ffffff;
        font-family: 'Arial Black', sans-serif;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    h2, h3 {
        color: #667eea;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>📈 Stock Price Prediction </h1>", unsafe_allow_html=True)
st.markdown("### Powered by Prophet, XGBoost, and LightGBM")

# Sidebar
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/machine-learning/machine-learning.png",
        width=100)
    st.title("⚙️ Settings")

    # Stock selector
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    selected_ticker = st.selectbox(
        "📊 Select Stock",
        tickers,
        help="Choose a stock to view predictions and analysis"
    )

    st.markdown("---")

    # Model selector
    models = ['LightGBM', 'XGBoost', 'Prophet']
    selected_model = st.selectbox(
        "🤖 Select Model",
        models,
        help="Choose ML model for comparison"
    )

    st.markdown("---")
    st.info("💡 **Tip:** LightGBM typically performs best!")


# Load data function
@st.cache_data
def load_stock_data(ticker):
    """Load processed stock data with features"""
    try:
        data_path = Path(f"demo_data/{ticker}_features.csv")
        if data_path.exists():
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data
def get_model_metrics():
    """Get model performance metrics"""
    # Simulated metrics (replace with actual MLflow data if needed)
    metrics = {
        'AAPL': {
            'Prophet': {'mape': 11.00, 'mae': 23.51, 'rmse': 28.74},
            'XGBoost': {'mape': 6.37, 'mae': 15.39, 'rmse': 19.45},
            'LightGBM': {'mape': 5.16, 'mae': 12.42, 'rmse': 15.90}
        },
        'GOOGL': {
            'Prophet': {'mape': 9.5, 'mae': 18.2, 'rmse': 22.3},
            'XGBoost': {'mape': 5.8, 'mae': 11.5, 'rmse': 14.2},
            'LightGBM': {'mape': 4.9, 'mae': 10.1, 'rmse': 12.8}
        },
        'MSFT': {
            'Prophet': {'mape': 10.2, 'mae': 20.5, 'rmse': 25.1},
            'XGBoost': {'mape': 6.1, 'mae': 13.8, 'rmse': 17.2},
            'LightGBM': {'mape': 5.3, 'mae': 11.9, 'rmse': 14.5}
        },
        'TSLA': {
            'Prophet': {'mape': 15.8, 'mae': 35.2, 'rmse': 42.5},
            'XGBoost': {'mape': 9.2, 'mae': 22.1, 'rmse': 28.3},
            'LightGBM': {'mape': 7.8, 'mae': 18.5, 'rmse': 23.7}
        },
        'AMZN': {
            'Prophet': {'mape': 11.5, 'mae': 25.3, 'rmse': 30.8},
            'XGBoost': {'mape': 6.8, 'mae': 16.2, 'rmse': 20.5},
            'LightGBM': {'mape': 5.7, 'mae': 13.8, 'rmse': 17.2}
        }
    }
    return metrics


# Load data
df = load_stock_data(selected_ticker)
metrics_data = get_model_metrics()

if df is not None:
    # Main content area
    col1, col2, col3, col4 = st.columns(4)

    # Latest price
    latest_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    price_change = ((latest_price - prev_price) / prev_price) * 100

    with col1:
        st.metric(
            label="💵 Latest Price",
            value=f"${latest_price:.2f}",
            delta=f"{price_change:.2f}%"
        )

    # Model metrics
    model_metrics = metrics_data[selected_ticker][selected_model]

    with col2:
        st.metric(
            label="🎯 MAPE",
            value=f"{model_metrics['mape']:.2f}%",
            delta=f"Accuracy: {100 - model_metrics['mape']:.2f}%"
        )

    with col3:
        st.metric(
            label="📊 MAE",
            value=f"${model_metrics['mae']:.2f}",
            delta="Lower is better"
        )

    with col4:
        st.metric(
            label="📈 RMSE",
            value=f"${model_metrics['rmse']:.2f}",
            delta="Root Mean Squared Error"
        )

    st.markdown("---")

    # Two columns for charts
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader(f"📉 {selected_ticker} Price History")

        # Price chart with moving averages
        fig = go.Figure()

        # Candlestick chart
        fig.add_trace(go.Scatter(
            x=df.index[-200:],
            y=df['Close'].iloc[-200:],
            mode='lines',
            name='Close Price',
            line=dict(color='#00D9FF', width=2)
        ))

        # Moving averages
        fig.add_trace(go.Scatter(
            x=df.index[-200:],
            y=df['SMA_50'].iloc[-200:],
            mode='lines',
            name='SMA 50',
            line=dict(color='#FF6B6B', width=1, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=df.index[-200:],
            y=df['SMA_200'].iloc[-200:],
            mode='lines',
            name='SMA 200',
            line=dict(color='#4ECDC4', width=1, dash='dash')
        ))

        fig.update_layout(
            template='plotly_dark',
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🏆 Model Comparison")

        # Model comparison chart
        comparison_df = pd.DataFrame({
            'Model': ['Prophet', 'XGBoost', 'LightGBM'],
            'MAPE': [
                metrics_data[selected_ticker]['Prophet']['mape'],
                metrics_data[selected_ticker]['XGBoost']['mape'],
                metrics_data[selected_ticker]['LightGBM']['mape']
            ]
        })

        fig_comparison = px.bar(
            comparison_df,
            x='Model',
            y='MAPE',
            color='MAPE',
            color_continuous_scale='Viridis',
            text='MAPE'
        )

        fig_comparison.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_comparison.update_layout(
            template='plotly_dark',
            height=400,
            showlegend=False,
            yaxis_title="MAPE (%)",
            xaxis_title=""
        )

        st.plotly_chart(fig_comparison, use_container_width=True)

    st.markdown("---")

    # Technical indicators
    col_tech1, col_tech2 = st.columns(2)

    with col_tech1:
        st.subheader("📊 Technical Indicators")

        # RSI Chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df.index[-100:],
            y=df['RSI'].iloc[-100:],
            mode='lines',
            name='RSI',
            line=dict(color='#FFD93D', width=2)
        ))

        # Overbought/Oversold lines
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")

        fig_rsi.update_layout(
            template='plotly_dark',
            height=300,
            title="RSI (Relative Strength Index)",
            yaxis_title="RSI",
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig_rsi, use_container_width=True)

    with col_tech2:
        st.subheader("📈 Volume Analysis")

        # Volume chart
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=df.index[-100:],
            y=df['Volume'].iloc[-100:],
            name='Volume',
            marker_color='#A8E6CF'
        ))

        fig_vol.update_layout(
            template='plotly_dark',
            height=300,
            title="Trading Volume",
            yaxis_title="Volume",
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")

    # Feature importance (simulated)
    st.subheader("🎯 Top Features Used by Model")

    feature_importance = {
        'LightGBM': {
            'Close_Lag_1': 437, 'Returns': 404, 'MACD_Hist': 134,
            'RSI': 128, 'EMA_12': 109, 'Volume_MA_20': 106
        },
        'XGBoost': {
            'EMA_12': 553, 'Close_Lag_1': 373, 'SMA_10': 52,
            'SMA_200': 31, 'BB_Middle': 25, 'Returns': 19
        },
        'Prophet': {
            'Trend': 600, 'Seasonality': 300, 'Holidays': 100
        }
    }

    importance_df = pd.DataFrame({
        'Feature': list(feature_importance[selected_model].keys()),
        'Importance': list(feature_importance[selected_model].values())
    })

    fig_importance = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Plasma'
    )

    fig_importance.update_layout(
        template='plotly_dark',
        height=300,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.plotly_chart(fig_importance, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #888;'>
            <p>Built with ❤️ using Streamlit • Data from Yahoo Finance • Models: Prophet, XGBoost, LightGBM</p>
            <p>📊 MLOps Pipeline: Airflow + MLflow + Docker</p>
        </div>
    """, unsafe_allow_html=True)

else:
    st.error(f"⚠️ No data found for {selected_ticker}. Please run the data pipeline first.")
    st.info("Run the Airflow DAGs to fetch and process data.")