# Stock Market Analysis and Prediction Using Neural Networks

A machine learning application that predicts stock prices using Recurrent Neural Networks (RNN). The application provides an interactive web interface built with Streamlit for analyzing and predicting stock prices of various Nepali banks.

## Features

- **Interactive Web Interface**: User-friendly Streamlit dashboard for stock price prediction
- **Multiple Bank Support**: Analyze stocks from different banks:
  - ADBL (Agriculture Development Bank Limited)
  - EBL (Everest Bank Limited)
  - HBL (Himalayan Bank Limited)
  - KBL (Kumari Bank Limited)
- **Real-time Predictions**: Visualize actual vs. predicted stock prices
- **Neural Network Models**: Pre-trained RNN models for accurate predictions
- **Data Visualization**: Clear graphical representation of stock price trends

## Technology Stack

- **Python 3.x**
- **Streamlit**: Web application framework
- **TensorFlow/Keras**: Deep learning framework for RNN models
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Scikit-learn**: Data preprocessing (MinMaxScaler)
- **PIL (Pillow)**: Image processing

## Project Structure

```
.
app.py                  # Main Streamlit application
models/                 # Trained neural network models
ADBL.h5            # ADBL stock prediction model
traning/               # Training and testing datasets
ADBL_train_data.csv
ADBL_test_data.csv
data/                  # Historical stock data (CSV files)
static/                # Static files for images
temp.png          # Generated prediction charts
merged.csv            # Consolidated stock data
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd STOCK-MARKET-PREDICTION-USING-RECURRENT-NEURAL-NETWORK
```

2. Install required dependencies:
```bash
pip install streamlit tensorflow pandas numpy matplotlib scikit-learn pillow
```

Or create a `requirements.txt` file with:
```
streamlit
tensorflow
pandas
numpy
matplotlib
scikit-learn
pillow
```

Then install:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Select a bank from the dropdown menu:
   - ADBL
   - EBL
   - HBL
   - KBL

4. Click the "ANALYZE" button to view the stock price prediction

5. The application will display a graph showing:
   - Red line: Actual stock prices
   - Blue line: Predicted stock prices

## How It Works

### Data Preprocessing
1. Historical stock data is loaded from CSV files
2. The 'Close' price column is extracted as the target variable
3. Data is normalized using MinMaxScaler (0-1 range)

### Model Architecture
- Recurrent Neural Network (RNN) trained on historical stock data
- Uses a 60-day window to predict future prices
- Models are pre-trained and stored in the `models/` directory

### Prediction Process
1. Load the pre-trained model for the selected bank
2. Prepare test data with proper scaling
3. Create sequences of 60 days for prediction
4. Generate predictions using the RNN model
5. Inverse transform predictions to original scale
6. Visualize results with matplotlib

## Model Details

- **Input Window**: 60 days of historical data
- **Prediction Window**: 20 days ahead
- **Normalization**: MinMaxScaler with feature range (0, 1)
- **Model Format**: Keras HDF5 (.h5)

## Data Format

Training and testing CSV files should contain the following columns:
- Date
- Open
- High
- Low
- Volume
- **Close** (primary feature for prediction)

## Future Enhancements

- [ ] Add more banks and stocks
- [ ] Implement real-time data fetching
- [ ] Add LSTM and GRU model variants
- [ ] Include technical indicators (RSI, MACD, etc.)
- [ ] Add model training interface
- [ ] Implement longer-term predictions
- [ ] Add confidence intervals for predictions
- [ ] Export predictions to CSV

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This application is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with a financial advisor before making investment choices.

## Acknowledgments

- Historical stock data sourced from Nepal Stock Exchange
- Built using open-source machine learning libraries
- Inspired by time series forecasting research
