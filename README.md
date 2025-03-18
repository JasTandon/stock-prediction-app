# Stock Prediction Application

This Python application uses machine learning to predict stock prices and provide day trading suggestions. Initially built with a basic LSTM model, it has evolved through multiple enhancements to include technical indicators, an interactive day trader mode, an expanded stock list, and corrected probability calculations. The current version, as of March 18, 2025, leverages historical data up to March 17, 2025, to predict closing prices for March 18, offering users actionable insights based on their investment goals.

## Features
- **Stock Price Prediction**: Uses an LSTM neural network with technical indicators (SMA, RSI, MACD) to predict next-day closing prices.
- **Interactive Day Trader Mode**: Prompts users for investment amount and target profit, suggesting stocks to maximize the chance of achieving the goal.
- **Expanded Stock List**: Covers 20 blue-chip and growth stocks (scalable to 100), including AAPL, GOOG, MSFT, AMZN, META, TSLA, and more.
- **Probability of Increase**: Calculates the likelihood of a price increase based on historical volatility, corrected for logical consistency.
- **Technical Indicators**: Enhances predictions with 5-day and 20-day SMA, 14-day RSI, and MACD (12, 26, 9).

## Changelog

### Initial Setup
- **Basic LSTM Model**: Predicted closing prices using historical OHLCV (Open, High, Low, Close, Volume) data for 5 stocks (AAPL, GOOG, MSFT, AMZN, META).
- **Data Source**: Fetched via `yfinance` from 2020-01-01 to March 17, 2025.
- **Output**: Simple table with predicted close, last open, change percentage, and probability.

### Enhancements
1. **Predicted Close Date (March 18, 2025)**:
    - Added "Predicted Close Date" column to clarify the prediction target (March 18, 2025, based on March 17 data).
    - Implemented `get_next_trading_day` to skip weekends, ensuring accurate trading day predictions.

2. **Technical Indicators (March 18, 2025)**:
    - Integrated `pandas_ta` to add SMA (5, 20), RSI (14), and MACD (12, 26, 9) as input features, expanding from 5 to 9 dimensions.
    - Fixed `NaN` import error in `pandas_ta` by patching `squeeze_pro.py` (`from numpy import nan as npNaN`).

3. **MultiIndex Fix (March 18, 2025)**:
    - Addressed `AttributeError` from `pandas_ta` due to `yfinance` MultiIndex columns by flattening to a simple Index in `fetch_data`.

4. **Interactive Day Trader Mode & More Stocks (March 18, 2025)**:
    - Added user input for investment amount and target profit, suggesting stocks meeting the required return.
    - Expanded stock list to 20 (e.g., TSLA, NVDA, JPM, BAC), aiming for 100 blue-chip/pink-chip stocks (sample provided).
    - Attempted real-time open price fetching, reverted to March 17 open as proxy due to `yfinance` limitations.

5. **Probability Correction (March 18, 2025)**:
    - Fixed illogical probabilities (e.g., AMZN: -2.57% change with 91.87% increase probability).
    - Corrected `z_score = (predicted_close - current_open) / (volatility * current_open)` and used `norm.cdf(z_score) * 100` for consistent increase probability.

## Prerequisites
- **Python**: 3.12+
- **Dependencies**:
    - `yfinance`
    - `pandas`
    - `pandas_ta`
    - `scikit-learn`
    - `tensorflow`
    - `numpy`
    - `tqdm`
    - `joblib`

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   python -m venv stocks_env
   source stocks_env/bin/activate  # On Windows: stocks_env\Scripts\activate

2. **Create Virtual Environment:**:
   ```bash
   git clone https://github.com/yourusername/stock-prediction-app.git
   cd stock-prediction-app

3. **Install Dependencies:**:
   ```bash
   pip install yfinance pandas pandas_ta scikit-learn tensorflow numpy tqdm joblib

4. **Patch pandas_ta (if needed):**:
   -  If you encounter ImportError: cannot import name 'NaN' from 'numpy':
      ```bash
      nano stocks_env/lib/python3.12/site-packages/pandas_ta/momentum/squeeze_pro.py
   -  Change from numpy import NaN as npNaN to from numpy import nan as npNaN.
   - Save and exit (Ctrl+O, Enter, Ctrl+X in nano).

5. **Directory Structure:**:
   ```bash
    stock-prediction-app/
    ├── main.py
    ├── utils.py
    ├── models/  # Created automatically for trained models
    ├── stocks_env/  # Virtual environment
    └── README.md

## Usage

1. **Activate Environment**:
   ```bash
   source stocks_env/bin/activate

2. **Run the Application**:
   - Training Phase: Trains LSTM models for each stock (saved in models/).
   - Prediction Phase: Generates predictions for March 18, 2025, using March 17 data.
   - Interactive Mode: Prompts for investment and target profit, suggesting stocks.
       ```bash
       python main.py
  
3. **Example Interaction:**:
   ```bash
    Starting training phase...
    [Training output...]
    
    Starting prediction phase...
    [Prediction output...]
    
    Stock Trading Suggestions (All Predictions):
    Stock | Last Open Date | Predicted Close Date | Predicted Close | Current Open | Change (%) | Probability of Increase (%)
    ------------------------------------------------------------------------------------------------------------------------
    ORCL  |   2025-03-17   |      2025-03-18      |       155.34 |    150.40 |      3.28 |      84.42
    ...
    
    === Day Trader Mode ===
    How much money do you want to invest today? (e.g., 1000): 1500
    What is your target profit for today? (e.g., 50): 150
    
    Required return to achieve $150.0 profit on $1500.0: 10.00%
    No stocks meet your target return of 10.00% today.
    Here are the best available options:
    Stock | Predicted Close Date | Predicted Close | Current Open | Expected Change (%) | Probability of Increase (%) | Expected Profit ($)
    --------------------------------------------------------------------------------------------------------------------------------------------
    ORCL  |      2025-03-18      |       155.34 |       150.40 |             3.28 |                  84.42 |              49.23
    ...
   

4. **Clone the Repository**:
   - To retrain models, clear the models/ directory: 
      ```bash
      rm -rf models/
      python main.py

## Limitations
    - Data Source: Uses yfinance, which lacks reliable real-time data for March 18, 2025 (proxies March 17 open prices). For true day trading, consider Alpha Vantage or IEX Cloud.
    - Accuracy: Predictions are based on historical patterns and technical indicators; market volatility and external factors (e.g., news) may reduce accuracy.
    - Scalability: Training 20 stocks takes ~10-20 minutes; scaling to 100 requires significant time or parallel processing (not implemented).
    - Quantum Inspiration: Requested quantum state/entanglement models weren’t implemented due to practical constraints; technical indicators serve as a complexity proxy.

## Future Enhancements
    - Real-Time Data: Integrate a real-time API (e.g., Alpha Vantage) for current open prices.
    - Ensemble Models: Add Random Forest or Gradient Boosting alongside LSTM for improved predictions.
    - Full Stock List: Expand to 100 blue-chip/pink-chip stocks with a comprehensive ticker list (e.g., S&P 500 + NASDAQ).
    - Risk Management: Incorporate portfolio diversification in day trader mode.

## Contributing
Feel free to fork this repository, submit issues, or pull requests to enhance functionality. Contributions to improve real-time data integration or add ensemble models are especially welcome!

## License
This project is licensed under the MIT License - see the LICENSE file for details (create one if desired).
