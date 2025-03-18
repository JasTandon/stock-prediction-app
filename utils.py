import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn import preprocessing
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import Callback
import joblib
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from datetime import datetime, timedelta

def fetch_data(stock, start_date, end_date):
    print(f"Fetching data for {stock} from {start_date} to {end_date}")
    data = yf.download(stock, start=start_date, end=end_date)
    if data.empty:
        print(f"Warning: No data retrieved for {stock}")
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    print(f"Retrieved {len(data)} days of data for {stock}")
    return data

def get_current_open(stock, data):
    """Use the last available open price as a proxy if current day data is unavailable."""
    if not data.empty and 'Open' in data.columns:
        return data['Open'].iloc[-1]
    print(f"Warning: No current open price for {stock}, using last available open as proxy")
    return None

def calculate_features(df):
    """Calculate technical indicators and return DataFrame with all features."""
    df = df.copy()
    df['SMA_5'] = ta.sma(df['Close'], length=5)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    macd_result = ta.macd(close=df['Close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd_result], axis=1)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9']
    return df[features].dropna()

def preprocess_data(data, sequence_length):
    print("Preprocessing data...")
    df = calculate_features(data)
    if len(df) < sequence_length + 1:
        print(f"Insufficient data: {len(df)} rows < {sequence_length + 1}")
        return None, None, None, None, None
    feature_scaler = preprocessing.MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(df.values)
    target_scaler = preprocessing.MinMaxScaler()
    target = df['Close'].values
    scaled_target = target_scaler.fit_transform(target[:, np.newaxis]).flatten()
    X = []
    y = []
    for i in tqdm(range(sequence_length, len(scaled_features)), desc="Creating sequences"):
        X.append(scaled_features[i - sequence_length:i])
        y.append(scaled_target[i])
    print(f"Created {len(X)} sequences")
    return np.array(X), np.array(y), feature_scaler, target_scaler, df

def train_model(X_train, y_train):
    print("Building LSTM regression model...")
    model = models.Sequential()
    model.add(layers.LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(layers.LSTM(50))
    model.add(layers.Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    print("Training model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, 
              callbacks=[TqdmCallback(verbose=1)])
    print("Model training completed")
    return model

def get_next_trading_day(current_date_str):
    """Calculate the next trading day, skipping weekends."""
    current_date = datetime.strptime(current_date_str, '%Y-%m-%d')
    next_day = current_date + timedelta(days=1)
    while next_day.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
        next_day += timedelta(days=1)
    return next_day.strftime('%Y-%m-%d')

def make_prediction(model, feature_scaler, target_scaler, data, sequence_length, stock):
    print(f"Preparing last sequence for prediction for {stock}...")
    df = calculate_features(data)
    last_sequence_df = df[-sequence_length:]
    last_sequence_scaled = feature_scaler.transform(last_sequence_df.values)
    X_pred = np.array([last_sequence_scaled])
    print(f"Making prediction for {stock}...")
    predicted_scaled_close = model.predict(X_pred, verbose=0)[0][0]
    predicted_close = target_scaler.inverse_transform([[predicted_scaled_close]])[0, 0]
    
    last_date = data.index[-1].strftime('%Y-%m-%d')
    pred_close_date = get_next_trading_day(last_date)
    
    current_open = get_current_open(stock, data)
    if current_open is None:
        print(f"Skipping {stock} due to missing current open price")
        return None, None, None, None, None
    
    recent_changes = df['Close'].pct_change().dropna()[-20:]
    volatility = recent_changes.std().item()
    if volatility == 0 or np.isnan(volatility):
        probability = 50.0
    else:
        # Corrected z-score: positive if increase, negative if decrease
        z_score = (predicted_close - current_open) / (volatility * current_open)
        probability = norm.cdf(z_score) * 100  # Probability of increase (close > open)
    
    return predicted_close, probability, current_open, last_date, pred_close_date

class TqdmCallback(Callback):
    def __init__(self, verbose=1):
        super().__init__()
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            desc = f"Epoch {epoch+1}/{self.epochs}"
            tqdm.write(f"{desc} - loss: {logs['loss']:.4f} - mae: {logs['mae']:.4f}")
