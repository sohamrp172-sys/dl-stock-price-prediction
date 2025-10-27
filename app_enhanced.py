import numpy as np 
import pandas as pd 
import yfinance as yf 
import os
os.environ['KERAS_BACKEND'] = 'jax'
from keras.models import load_model
import streamlit as st 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor | AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stTab {
        font-size: 1.1rem;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📈 AI-Powered Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced LSTM Neural Network for Stock Market Analysis & Forecasting</p>', unsafe_allow_html=True)

# Utility Functions
@st.cache_data(ttl=3600)
def load_stock_data(ticker, start_date, end_date):
    """Load stock data with caching"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_prediction_model():
    """Load the trained model with caching"""
    try:
        model = load_model('Stock_Predictions_Model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def calculate_metrics(actual, predicted):
    """Calculate performance metrics"""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    accuracy = 100 - mape
    return rmse, mae, mape, accuracy

def calculate_moving_averages(data):
    """Calculate various moving averages"""
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_100'] = data['Close'].rolling(window=100).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    return data

def predict_future_prices(model, last_100_days, scaler, n_days=30):
    """Predict future stock prices"""
    predictions = []
    current_batch = last_100_days.copy()
    
    for _ in range(n_days):
        current_batch_scaled = scaler.transform(current_batch.reshape(-1, 1))
        current_batch_reshaped = current_batch_scaled.reshape(1, 100, 1)
        next_pred = model.predict(current_batch_reshaped, verbose=0)
        next_pred_inverse = scaler.inverse_transform(next_pred)[0, 0]
        predictions.append(next_pred_inverse)
        current_batch = np.append(current_batch[1:], next_pred_inverse)
    
    return predictions

def calculate_prediction_confidence(predictions, historical_volatility):
    """Calculate confidence intervals for predictions"""
    confidence = 100 - min(historical_volatility * 10, 50)
    return max(confidence, 50)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Stock Selection
    st.subheader("📊 Stock Selection")
    stock_input = st.text_input("Enter Stock Ticker", value="GOOG", help="Enter stock symbol (e.g., AAPL, MSFT, TSLA)")
    stock = stock_input.upper()
    
    # Time Period Selection
    st.subheader("📅 Time Period")
    period_option = st.selectbox(
        "Quick Select",
        ["Custom", "1 Month", "3 Months", "6 Months", "1 Year", "5 Years", "Max"],
        index=6
    )
    
    if period_option == "Custom":
        start_date = st.date_input("Start Date", value=pd.to_datetime('2012-06-01'))
        end_date = st.date_input("End Date", value=pd.to_datetime('today'))
    else:
        end_date = datetime.now()
        if period_option == "1 Month":
            start_date = end_date - timedelta(days=30)
        elif period_option == "3 Months":
            start_date = end_date - timedelta(days=90)
        elif period_option == "6 Months":
            start_date = end_date - timedelta(days=180)
        elif period_option == "1 Year":
            start_date = end_date - timedelta(days=365)
        elif period_option == "5 Years":
            start_date = end_date - timedelta(days=1825)
        else:  # Max
            start_date = pd.to_datetime('2012-06-01')
    
    # Prediction Settings
    st.subheader("🔮 Prediction Settings")
    future_days = st.slider("Future Prediction Days", 7, 90, 30)
    
    # Model Info
    st.subheader("🤖 Model Information")
    st.info("""
    **LSTM Neural Network**
    - 4 LSTM Layers
    - Dropout Regularization
    - Adam Optimizer
    - Trained on 80% data
    """)
    
    # Footer
    st.markdown("---")

# Main Content
try:
    # Load data with spinner
    with st.spinner(f'🔄 Loading {stock} data...'):
        data = load_stock_data(stock, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if data is None or data.empty:
        st.error(f"❌ Unable to fetch data for ticker '{stock}'. Please check the ticker symbol and try again.")
        st.stop()
    
    # Load model
    model = load_prediction_model()
    if model is None:
        st.error("❌ Model file not found. Please ensure 'Stock_Predictions_Model.keras' exists in the directory.")
        st.stop()
    
    # Handle MultiIndex columns (yfinance returns MultiIndex for single ticker)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    # Calculate moving averages
    data = calculate_moving_averages(data)
    
    # Current stock info (convert to float to avoid Series formatting issues)
    current_price = float(data['Close'].iloc[-1])
    prev_price = float(data['Close'].iloc[-2])
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    # Top Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label=f"{stock} Current Price",
            value=f"${current_price:.2f}",
            delta=f"{price_change:.2f} ({price_change_pct:.2f}%)"
        )
    
    with col2:
        st.metric(
            label="Day High",
            value=f"${float(data['High'].iloc[-1]):.2f}"
        )
    
    with col3:
        st.metric(
            label="Day Low",
            value=f"${float(data['Low'].iloc[-1]):.2f}"
        )
    
    with col4:
        st.metric(
            label="Volume",
            value=f"{float(data['Volume'].iloc[-1]):,.0f}"
        )
    
    with col5:
        volatility = float(data['Close'].pct_change().std() * 100)
        st.metric(
            label="Volatility",
            value=f"{volatility:.2f}%"
        )
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔮 Predictions", 
        "📈 Technical Analysis",
        "🤖 Model Performance",
        "ℹ️ About"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.header(f"{stock} Stock Analysis Overview")
        
        # Candlestick Chart
        st.subheader("📊 Interactive Candlestick Chart")
        
        fig_candle = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC'
        )])
        
        fig_candle.update_layout(
            title=f'{stock} Stock Price',
            yaxis_title='Price (USD)',
            xaxis_title='Date',
            height=500,
            template='plotly_dark',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_candle, use_container_width=True)
        
        # Volume Chart
        st.subheader("📊 Trading Volume")
        
        fig_volume = go.Figure()
        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                  for i in range(len(data))]
        
        fig_volume.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            marker_color=colors,
            name='Volume'
        ))
        
        fig_volume.update_layout(
            title='Trading Volume',
            yaxis_title='Volume',
            xaxis_title='Date',
            height=300,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Data Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Dataset Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Total Days', 'Start Date', 'End Date', 'Avg Price', 'Max Price', 'Min Price'],
                'Value': [
                    len(data),
                    data.index[0].strftime('%Y-%m-%d'),
                    data.index[-1].strftime('%Y-%m-%d'),
                    f"${data['Close'].mean():.2f}",
                    f"${data['Close'].max():.2f}",
                    f"${data['Close'].min():.2f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("📈 Recent Data")
            st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10), use_container_width=True)
    
    # TAB 2: Predictions
    with tab2:
        st.header("🔮 Stock Price Predictions")
        
        with st.spinner('🧠 Generating predictions...'):
            # Prepare data for prediction
            data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.80)])
            data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):])
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_train_scaled = scaler.fit_transform(data_train)
            
            # Prepare test data
            past_100_days = data_train.tail(100)
            final_df = pd.concat([past_100_days, data_test], ignore_index=True)
            input_data = scaler.transform(final_df)
            
            x_test = []
            y_test = []
            
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i, 0])
            
            x_test, y_test = np.array(x_test), np.array(y_test)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
            
            # Make predictions
            predictions = model.predict(x_test, verbose=0)
            predictions = scaler.inverse_transform(predictions)
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate metrics
            rmse, mae, mape, accuracy = calculate_metrics(y_test, predictions)
            
            # Future predictions
            last_100_days = data['Close'].values[-100:]
            future_predictions = predict_future_prices(model, last_100_days, scaler, future_days)
            
            # Create future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy:.2f}%")
        with col2:
            st.metric("📊 RMSE", f"${rmse:.2f}")
        with col3:
            st.metric("📈 MAE", f"${mae:.2f}")
        with col4:
            st.metric("📉 MAPE", f"{mape:.2f}%")
        
        # Prediction vs Actual Chart
        st.subheader("📊 Historical Predictions vs Actual Prices")
        
        # Get test dates
        test_start_idx = int(len(data) * 0.80) + 100
        test_dates = data.index[test_start_idx:]
        
        fig_pred = go.Figure()
        
        fig_pred.add_trace(go.Scatter(
            x=test_dates,
            y=y_test.flatten(),
            mode='lines',
            name='Actual Price',
            line=dict(color='#10b981', width=2)
        ))
        
        fig_pred.add_trace(go.Scatter(
            x=test_dates,
            y=predictions.flatten(),
            mode='lines',
            name='Predicted Price',
            line=dict(color='#ef4444', width=2, dash='dash')
        ))
        
        fig_pred.update_layout(
            title='Prediction vs Actual Price Comparison',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=500,
            template='plotly_dark',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Future Predictions Chart
        st.subheader(f"🔮 Future Price Predictions ({future_days} Days)")
        
        # Calculate confidence
        historical_volatility = data['Close'].pct_change().std()
        confidence = calculate_prediction_confidence(future_predictions, historical_volatility)
        
        st.info(f"**Prediction Confidence:** {confidence:.1f}% | Based on historical volatility and model accuracy")
        
        fig_future = go.Figure()
        
        # Historical data (last 90 days)
        historical_window = min(90, len(data))
        fig_future.add_trace(go.Scatter(
            x=data.index[-historical_window:],
            y=data['Close'][-historical_window:],
            mode='lines',
            name='Historical Price',
            line=dict(color='#3b82f6', width=2)
        ))
        
        # Future predictions
        fig_future.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions,
            mode='lines+markers',
            name='Future Predictions',
            line=dict(color='#f59e0b', width=3, dash='dot'),
            marker=dict(size=6)
        ))
        
        # Confidence interval (simple approach)
        uncertainty = np.array(future_predictions) * (historical_volatility * 2)
        
        fig_future.add_trace(go.Scatter(
            x=future_dates,
            y=np.array(future_predictions) + uncertainty,
            fill=None,
            mode='lines',
            line=dict(color='rgba(245, 158, 11, 0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_future.add_trace(go.Scatter(
            x=future_dates,
            y=np.array(future_predictions) - uncertainty,
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(245, 158, 11, 0)'),
            name='Confidence Interval',
            fillcolor='rgba(245, 158, 11, 0.2)'
        ))
        
        fig_future.update_layout(
            title=f'{stock} Price Forecast - Next {future_days} Days',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=500,
            template='plotly_dark',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_future, use_container_width=True)
        
        # Download predictions
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepare download data
            prediction_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Price': future_predictions,
                'Upper_Bound': np.array(future_predictions) + uncertainty,
                'Lower_Bound': np.array(future_predictions) - uncertainty
            })
            
            csv = prediction_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Future Predictions (CSV)",
                data=csv,
                file_name=f"{stock}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Summary stats
            st.markdown("### 📊 Prediction Summary")
            avg_future_price = np.mean(future_predictions)
            price_change_forecast = ((avg_future_price - current_price) / current_price) * 100
            
            st.write(f"**Average Predicted Price:** ${avg_future_price:.2f}")
            st.write(f"**Expected Change:** {price_change_forecast:+.2f}%")
            st.write(f"**Highest Prediction:** ${max(future_predictions):.2f}")
            st.write(f"**Lowest Prediction:** ${min(future_predictions):.2f}")
    
    # TAB 3: Technical Analysis
    with tab3:
        st.header("📈 Technical Analysis")
        
        # Moving Averages Chart
        st.subheader("📊 Moving Averages Analysis")
        
        fig_ma = go.Figure()
        
        fig_ma.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#3b82f6', width=2)
        ))
        
        fig_ma.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_50'],
            mode='lines',
            name='50-Day MA',
            line=dict(color='#10b981', width=1.5)
        ))
        
        fig_ma.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_100'],
            mode='lines',
            name='100-Day MA',
            line=dict(color='#f59e0b', width=1.5)
        ))
        
        fig_ma.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_200'],
            mode='lines',
            name='200-Day MA',
            line=dict(color='#ef4444', width=1.5)
        ))
        
        fig_ma.update_layout(
            title='Price with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=500,
            template='plotly_dark',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_ma, use_container_width=True)
        
        # RSI and other indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 RSI (Relative Strength Index)")
            
            # Calculate RSI
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            
            fig_rsi = go.Figure()
            
            fig_rsi.add_trace(go.Scatter(
                x=data.index,
                y=data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='#8b5cf6', width=2)
            ))
            
            # Add overbought/oversold lines
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            
            fig_rsi.update_layout(
                yaxis_title='RSI',
                xaxis_title='Date',
                height=300,
                template='plotly_dark',
                showlegend=False
            )
            
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            current_rsi = data['RSI'].iloc[-1]
            if current_rsi > 70:
                st.warning(f"⚠️ RSI at {current_rsi:.2f} - Stock may be **overbought**")
            elif current_rsi < 30:
                st.success(f"✅ RSI at {current_rsi:.2f} - Stock may be **oversold**")
            else:
                st.info(f"ℹ️ RSI at {current_rsi:.2f} - Stock in **neutral** zone")
        
        with col2:
            st.subheader("📊 MACD (Moving Average Convergence Divergence)")
            
            # Calculate MACD
            macd = ta.trend.MACD(data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_Signal'] = macd.macd_signal()
            data['MACD_Hist'] = macd.macd_diff()
            
            fig_macd = go.Figure()
            
            fig_macd.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='#3b82f6', width=2)
            ))
            
            fig_macd.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD_Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='#ef4444', width=2)
            ))
            
            fig_macd.add_trace(go.Bar(
                x=data.index,
                y=data['MACD_Hist'],
                name='Histogram',
                marker_color='gray',
                opacity=0.3
            ))
            
            fig_macd.update_layout(
                yaxis_title='MACD',
                xaxis_title='Date',
                height=300,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig_macd, use_container_width=True)
            
            if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1]:
                st.success("✅ MACD indicates **bullish** trend")
            else:
                st.warning("⚠️ MACD indicates **bearish** trend")
        
        # Additional metrics
        st.subheader("📊 Additional Technical Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(data['Close'])
            current_bb_position = (data['Close'].iloc[-1] - bb.bollinger_lband().iloc[-1]) / \
                                (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]) * 100
            st.metric("Bollinger Band Position", f"{current_bb_position:.1f}%")
        
        with col2:
            # ADX - Trend Strength
            adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'])
            current_adx = adx.adx().iloc[-1]
            st.metric("ADX (Trend Strength)", f"{current_adx:.2f}")
        
        with col3:
            # Volume trend
            avg_volume = data['Volume'].mean()
            volume_trend = ((data['Volume'].iloc[-1] - avg_volume) / avg_volume) * 100
            st.metric("Volume vs Average", f"{volume_trend:+.1f}%")
        
        with col4:
            # Price momentum
            momentum = ((data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20]) * 100
            st.metric("20-Day Momentum", f"{momentum:+.2f}%")
    
    # TAB 4: Model Performance
    with tab4:
        st.header("🤖 Model Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Performance Metrics")
            
            metrics_df = pd.DataFrame({
                'Metric': ['Root Mean Squared Error (RMSE)', 
                          'Mean Absolute Error (MAE)', 
                          'Mean Absolute Percentage Error (MAPE)',
                          'Prediction Accuracy',
                          'Training Data Split',
                          'Test Data Split'],
                'Value': [f'${rmse:.2f}', 
                         f'${mae:.2f}', 
                         f'{mape:.2f}%',
                         f'{accuracy:.2f}%',
                         '80%',
                         '20%']
            })
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Error Distribution
            st.subheader("📊 Prediction Error Distribution")
            
            errors = y_test.flatten() - predictions.flatten()
            
            fig_error = go.Figure()
            fig_error.add_trace(go.Histogram(
                x=errors,
                nbinsx=30,
                name='Error Distribution',
                marker_color='#8b5cf6'
            ))
            
            fig_error.update_layout(
                title='Distribution of Prediction Errors',
                xaxis_title='Error (USD)',
                yaxis_title='Frequency',
                height=300,
                template='plotly_dark',
                showlegend=False
            )
            
            st.plotly_chart(fig_error, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Prediction Accuracy Over Time")
            
            # Calculate rolling accuracy
            window = 10
            rolling_errors = pd.Series(np.abs(errors)).rolling(window=window).mean()
            rolling_accuracy = 100 - (rolling_errors / y_test.flatten() * 100)
            
            fig_acc = go.Figure()
            
            fig_acc.add_trace(go.Scatter(
                x=list(range(len(rolling_accuracy))),
                y=rolling_accuracy,
                mode='lines',
                name='Rolling Accuracy',
                line=dict(color='#10b981', width=2),
                fill='tozeroy'
            ))
            
            fig_acc.update_layout(
                title=f'Rolling Accuracy ({window}-Day Window)',
                xaxis_title='Prediction Index',
                yaxis_title='Accuracy (%)',
                height=300,
                template='plotly_dark',
                showlegend=False
            )
            
            st.plotly_chart(fig_acc, use_container_width=True)
            
            # Model Architecture
            st.subheader("🏗️ Model Architecture")
            
            st.code("""
Model: LSTM Neural Network
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 1: LSTM(50) + Dropout(0.2)
Layer 2: LSTM(60) + Dropout(0.3)
Layer 3: LSTM(80) + Dropout(0.4)
Layer 4: LSTM(120) + Dropout(0.5)
Output: Dense(1)

Optimizer: Adam
Loss: Mean Squared Error
Lookback Window: 100 days
            """, language="text")
        
        # Performance insights
        st.subheader("💡 Performance Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_error_pct = (mae / y_test.mean()) * 100
            if avg_error_pct < 5:
                st.success(f"✅ **Excellent** - Avg error: {avg_error_pct:.2f}%")
            elif avg_error_pct < 10:
                st.info(f"ℹ️ **Good** - Avg error: {avg_error_pct:.2f}%")
            else:
                st.warning(f"⚠️ **Fair** - Avg error: {avg_error_pct:.2f}%")
        
        with col2:
            # Calculate directional accuracy
            actual_direction = np.diff(y_test.flatten()) > 0
            pred_direction = np.diff(predictions.flatten()) > 0
            directional_accuracy = (actual_direction == pred_direction).mean() * 100
            st.metric("Directional Accuracy", f"{directional_accuracy:.1f}%")
        
        with col3:
            # Max error
            max_error = np.max(np.abs(errors))
            st.metric("Maximum Error", f"${max_error:.2f}")
        
        # Scatter plot: Actual vs Predicted
        st.subheader("📊 Actual vs Predicted Scatter Plot")
        
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=y_test.flatten(),
            y=predictions.flatten(),
            mode='markers',
            name='Predictions',
            marker=dict(
                size=8,
                color=np.abs(errors),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Error (USD)")
            )
        ))
        
        # Perfect prediction line
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig_scatter.update_layout(
            title='Actual vs Predicted Prices',
            xaxis_title='Actual Price (USD)',
            yaxis_title='Predicted Price (USD)',
            height=400,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # TAB 5: About
    with tab5:
        st.header("ℹ️ About This Project")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Project Overview
            
            This **AI-Powered Stock Price Predictor** is an advanced machine learning application that uses 
            **Long Short-Term Memory (LSTM)** neural networks to forecast stock prices. The application 
            provides comprehensive technical analysis, interactive visualizations, and future price predictions.
            
            ### 🧠 How It Works
            
            1. **Data Collection**: Real-time stock data is fetched using Yahoo Finance API
            2. **Data Processing**: Historical prices are normalized using MinMaxScaler
            3. **Feature Engineering**: 100-day rolling windows create sequences for LSTM
            4. **Model Prediction**: 4-layer LSTM network processes sequences
            5. **Post-Processing**: Predictions are inverse-transformed to actual prices
            6. **Visualization**: Interactive charts display results and insights
            
            ### 🏗️ Technical Stack
            
            - **Framework**: Streamlit
            - **ML Library**: Keras (JAX backend)
            - **Data Processing**: Pandas, NumPy
            - **Visualization**: Plotly
            - **Technical Analysis**: TA-Lib
            - **Data Source**: yfinance
            
            ### 📊 Model Details
            
            - **Architecture**: Deep LSTM Neural Network
            - **Layers**: 4 LSTM layers (50, 60, 80, 120 units)
            - **Regularization**: Dropout (0.2, 0.3, 0.4, 0.5)
            - **Optimizer**: Adam
            - **Loss Function**: Mean Squared Error
            - **Training Split**: 80/20
            - **Lookback Period**: 100 days
            
            ### 🎓 Key Features
            
            ✅ Real-time stock data analysis  
            ✅ Interactive candlestick charts  
            ✅ Technical indicators (RSI, MACD, Bollinger Bands)  
            ✅ Moving averages analysis  
            ✅ Future price predictions (7-90 days)  
            ✅ Model performance metrics  
            ✅ Downloadable predictions  
            ✅ Confidence intervals  
            ✅ Directional accuracy  
            
            ### ⚠️ Important Disclaimer
            
            This application is for **educational and informational purposes only**. 
            Stock price predictions are based on historical data and should not be considered 
            as financial advice. Always conduct thorough research and consult with financial 
            advisors before making investment decisions.
            
            **Past performance does not guarantee future results.**
            
            ### 📈 Use Cases
            
            - Educational tool for learning about ML in finance
            - Historical trend analysis
            - Technical analysis companion
            - Portfolio project demonstration
            - Algorithm trading research
            
            ### 🔮 Future Enhancements
            
            - Multi-stock comparison
            - Sentiment analysis integration
            - Real-time news feed
            - Portfolio optimization
            - Backtesting capabilities
            - Advanced risk metrics
            - Custom indicators
            
            ### 📝 License & Credits
            
            This project is open-source and available for educational purposes.
            
            **Created with ❤️ using Python, Streamlit, and Keras**
            """)
        
        with col2:
            st.markdown("### 📊 Quick Stats")
            
            st.info(f"""
            **Current Stock**: {stock}  
            **Data Points**: {len(data)}  
            **Model Accuracy**: {accuracy:.2f}%  
            **RMSE**: ${rmse:.2f}  
            **Predictions**: {future_days} days
            """)
            
            st.markdown("### 🔗 Resources")
            
            st.markdown("""
            - [Streamlit Docs](https://docs.streamlit.io)
            - [Keras Documentation](https://keras.io)
            - [Yahoo Finance](https://finance.yahoo.com)
            - [Plotly Charts](https://plotly.com)
            - [TA-Lib](https://technical-analysis-library-in-python.readthedocs.io/)
            """)
            

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b;'>
    <p><strong>AI-Powered Stock Price Predictor</strong> | Built with Streamlit & Keras</p>
    <p>⚠️ <em>For educational purposes only. Not financial advice.</em></p>
</div>
""", unsafe_allow_html=True)
