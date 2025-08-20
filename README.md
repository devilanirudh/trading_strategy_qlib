# 🔬 Qlib Intelligent Trading Dashboard

A modular, FastAPI-based quantitative trading system powered by Microsoft Qlib, featuring advanced alpha generation, risk management, and real-time web dashboard.

## 🏗️ Architecture

The system is organized into modular components:

```
qlib/
├── main.py                 # Main FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── models/
│   ├── __init__.py
│   └── alpha_engine.py    # Core quantitative analysis engine
├── routers/
│   ├── __init__.py
│   ├── analysis.py        # API endpoints for analysis
│   └── views.py           # HTML pages and views
├── utils/
│   ├── __init__.py
│   └── chart_utils.py     # Plotly chart utilities
├── static/                # Static files (CSS, JS, images)
└── templates/             # HTML templates
```

## 🚀 Features

### Quantitative Analysis
- **Alpha Factors**: 10+ sophisticated alpha factors (momentum, mean reversion, volatility, volume, microstructure)
- **Market Regime Detection**: ML-based regime classification (Normal, High Volatility, Trending, Stressed)
- **Risk Management**: Comprehensive risk metrics (VaR, CVaR, Max Drawdown, Sharpe, Calmar, Sortino ratios)
- **Portfolio Optimization**: Kelly Criterion, volatility scaling, dynamic position sizing
- **Stop Loss & Take Profit**: Regime-dependent SL/TP calculation

### Web Interface
- **Interactive Dashboard**: Real-time charts with Plotly
- **CSV Upload**: Drag-and-drop CSV file analysis
- **Performance Metrics**: Live performance tracking
- **Trading Signals**: Current signal with position details
- **API Endpoints**: RESTful API for programmatic access

## 📦 Installation

1. **Clone and navigate to the qlib directory:**
```bash
cd qlib
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python main.py
```

4. **Access the dashboard:**
- Web Dashboard: http://localhost:8080
- API Documentation: http://localhost:8080/docs
- Health Check: http://localhost:8080/health

## 📊 Usage

### Web Dashboard
1. Open http://localhost:8080
2. Upload your CSV file with OHLCV data
3. View real-time analysis results
4. Monitor trading signals and performance metrics

### API Endpoints

#### POST `/api/analyze`
Upload and analyze CSV data:
```bash
curl -X POST "http://localhost:8080/api/analyze" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_data.csv"
```

#### GET `/api/dashboard`
Get current dashboard data:
```bash
curl "http://localhost:8080/api/dashboard"
```

#### GET `/api/signal`
Get current trading signal:
```bash
curl "http://localhost:8080/api/signal"
```

#### GET `/api/status`
Get system status:
```bash
curl "http://localhost:8080/api/status"
```

## 📁 CSV Format

Your CSV file should have the following columns:
```csv
Date Time,Open,High,Low,Close,Volume
2023-01-01 09:15:00,100.0,101.0,99.0,100.5,1000
2023-01-01 09:16:00,100.5,102.0,100.0,101.5,1200
...
```

## 🔧 Configuration

### Alpha Factors
The system computes 10+ alpha factors:
- **Momentum**: Multi-timeframe momentum (5min, 15min, 30min, 60min)
- **Mean Reversion**: Distance from moving averages (10, 30, 60 periods)
- **Volatility**: Realized volatility with multiple windows
- **Volume**: Volume momentum and price-volume correlation
- **Microstructure**: High-low spread, price efficiency
- **Technical**: RSI, Bollinger Bands, MACD
- **Regime**: Market stress indicators, trend strength

### Risk Management
- **Position Sizing**: Kelly Criterion with volatility scaling
- **Stop Loss**: Dynamic SL based on volatility and market regime
- **Take Profit**: Regime-dependent TP levels
- **Risk Per Trade**: 1% of account size (configurable)

## 🎯 Trading Signals

The system generates three types of signals:
- **BUY**: Positive alpha signal with long position
- **SELL**: Negative alpha signal with short position
- **HOLD**: Neutral signal, stay flat

Each signal includes:
- Signal strength and confidence
- Position size and value
- Stop loss and take profit levels
- Risk-reward ratio
- Maximum loss and profit

## 📈 Performance Metrics

- **Total Return**: Overall strategy performance
- **Annual Return**: Annualized return rate
- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Hit Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss ratio

## 🔄 Real-time Integration

The system is designed for real-time trading:
- WebSocket-ready architecture
- State management for live data
- Real-time signal generation
- Live performance tracking

## 🛠️ Development

### Adding New Alpha Factors
1. Edit `models/alpha_engine.py`
2. Add factor computation in `compute_alpha_factors()`
3. Include in `alpha_factors` dictionary
4. Update regime weights if needed

### Adding New API Endpoints
1. Edit `routers/analysis.py`
2. Add new endpoint function
3. Include proper error handling
4. Update documentation

### Customizing Charts
1. Edit `utils/chart_utils.py`
2. Modify chart creation functions
3. Update web interface in `routers/views.py`

## 📝 License

This project is for educational and research purposes. Use at your own risk.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the API documentation at http://localhost:8080/docs
- Review the health check at http://localhost:8080/health
- Examine the system logs for error details
# trading_strategy_qlib
# trading_strategy_qlib
