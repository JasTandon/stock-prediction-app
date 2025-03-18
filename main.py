import os
import joblib
from datetime import datetime, timedelta
from utils import fetch_data, preprocess_data, train_model, make_prediction
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras import models

stocks = [
    'AAPL', 'GOOG', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'BAC', 'WMT',
    'XOM', 'CVX', 'PG', 'KO', 'PEP', 'DIS', 'CSCO', 'INTC', 'AMD', 'ORCL'
]

start_date = '2020-01-01'
sequence_length = 20

today = datetime.now().date()
current_time_pdt = datetime.now().strftime('%H:%M')
market_close_pdt = '13:00'
if current_time_pdt < market_close_pdt:
    end_date_predict = (today - timedelta(days=1)).strftime('%Y-%m-%d')  # March 17, 2025
    end_date_train = (datetime.fromisoformat(end_date_predict) - timedelta(days=2)).strftime('%Y-%m-%d')  # March 15, 2025
else:
    end_date_predict = today.strftime('%Y-%m-%d')  # March 18, 2025
    end_date_train = (today - timedelta(days=1)).strftime('%Y-%m-%d')  # March 17, 2025

def train():
    for stock in tqdm(stocks, desc="Training stocks"):
        print(f"\nProcessing {stock}")
        data = fetch_data(stock, start_date, end_date_train)
        if data.empty:
            print(f"Skipping {stock} due to no data")
            continue
        X, y, feature_scaler, target_scaler, df = preprocess_data(data, sequence_length)
        if X is None:
            print(f"Skipping {stock} due to insufficient data")
            continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = train_model(X_train, y_train)
        model_path = f"models/{stock}/model.keras"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        joblib.dump(feature_scaler, f"models/{stock}/feature_scaler.joblib")
        joblib.dump(target_scaler, f"models/{stock}/target_scaler.joblib")
        print(f"Saved model and scalers for {stock}")

def predict():
    predictions = {}
    for stock in tqdm(stocks, desc="Predicting stocks"):
        print(f"\nPredicting for {stock}")
        data = fetch_data(stock, start_date, end_date_predict)
        if data.empty:
            print(f"Skipping {stock} due to no data")
            continue
        model_path = f"models/{stock}/model.keras"
        if not os.path.exists(model_path):
            print(f"No trained model found for {stock}, skipping")
            continue
        model = models.load_model(model_path)
        feature_scaler = joblib.load(f"models/{stock}/feature_scaler.joblib")
        target_scaler = joblib.load(f"models/{stock}/target_scaler.joblib")
        predicted_close, probability, current_open, last_date, pred_close_date = make_prediction(
            model, feature_scaler, target_scaler, data, sequence_length, stock
        )
        if predicted_close is None:
            continue
        predictions[stock] = (predicted_close, probability, current_open, last_date, pred_close_date)
    return predictions

def interactive_mode(predictions):
    print("\n=== Day Trader Mode ===")
    try:
        investment = float(input("How much money do you want to invest today? (e.g., 1000): "))
        target_profit = float(input("What is your target profit for today? (e.g., 50): "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return
    
    required_return = (target_profit / investment) * 100
    print(f"\nRequired return to achieve ${target_profit} profit on ${investment}: {required_return:.2f}%")
    
    suggestions = []
    for stock, (pred_close, prob, curr_open, last_date, pred_close_date) in predictions.items():
        expected_change = ((pred_close - curr_open) / curr_open) * 100
        expected_profit = (expected_change / 100) * investment
        if expected_change >= required_return:
            suggestions.append((stock, pred_close, curr_open, expected_change, prob, expected_profit))
    
    if not suggestions:
        print(f"\nNo stocks meet your target return of {required_return:.2f}% today.")
        print("Here are the best available options:")
        all_options = [(stock, pred_close, curr_open, ((pred_close - curr_open) / curr_open) * 100, prob)
                       for stock, (pred_close, prob, curr_open, _, _) in predictions.items()]
        suggestions = sorted(all_options, key=lambda x: x[3], reverse=True)[:5]  # Top 5 by change
    
    print("\nSuggested Stocks for Day Trading:")
    print("Stock | Predicted Close Date | Predicted Close | Current Open | Expected Change (%) | Probability of Increase (%) | Expected Profit ($)")
    print("-" * 140)
    for stock, pred_close, curr_open, change_pct, prob, *extra in suggestions:
        expected_profit = extra[0] if len(extra) > 0 else ((pred_close - curr_open) / curr_open) * investment
        print(f"{stock:<5} | {pred_close_date:^20} | {pred_close:>12.2f} | {curr_open:>12.2f} | {change_pct:>16.2f} | {prob:>22.2f} | {expected_profit:>18.2f}")

# Main execution
print("Starting training phase...")
train()

print("\nStarting prediction phase...")
predictions = predict()

print("\nStock Trading Suggestions (All Predictions):")
print("Stock | Last Open Date | Predicted Close Date | Predicted Close | Current Open | Change (%) | Probability of Increase (%)")
print("-" * 120)
for stock, (pred_close, prob, curr_open, last_date, pred_close_date) in sorted(predictions.items(), key=lambda x: x[1][1], reverse=True):
    change_pct = ((pred_close - curr_open) / curr_open) * 100
    print(f"{stock:<5} | {last_date:^14} | {pred_close_date:^20} | {pred_close:>12.2f} | {curr_open:>9.2f} | {change_pct:>9.2f} | {prob:>10.2f}")

interactive_mode(predictions)
