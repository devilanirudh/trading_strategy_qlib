#!/usr/bin/env python3
"""
Quantitative Alpha Engine - Core trading analysis engine with Qlib Alpha datasets
"""

import pandas as pd
import numpy as np
import qlib
from qlib.config import REG_CN
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Qlib imports
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158, Alpha360

class QuantitativeAlphaEngine:
    """
    Advanced Quantitative Alpha Engine using Qlib's capabilities and built-in Alpha datasets
    """
    
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = None
        self.alpha_factors = {}
        self.regime_states = None
        self.risk_metrics = {}
        self.portfolio_weights = None
        self.backtest_results = None
        self.qlib_alpha_factors = {}
        self.comparison_metrics = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data with Qlib Alpha features"""
        print("📊 Loading and preparing data with Qlib Alpha features...")
        
        # Load basic CSV data
        df = pd.read_csv(self.csv_file)
        df['Date Time'] = pd.to_datetime(df['Date Time'])
        df.set_index('Date Time', inplace=True)
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Basic returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']).diff()
        
        # Volume-weighted average price
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Remove outliers using IQR method
        Q1 = df['returns'].quantile(0.25)
        Q3 = df['returns'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['returns'] < (Q1 - 1.5 * IQR)) | (df['returns'] > (Q3 + 1.5 * IQR)))]
        
        # Clean infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remove NaN values
        df.dropna(inplace=True)
        
        # Add Qlib Alpha features
        df = self.add_qlib_alpha_features(df)
        
        # Final data cleaning
        df = df.replace([np.inf, -np.inf], np.nan)
        df.dropna(inplace=True)
        
        self.data = df
        print(f"✅ Data prepared: {df.shape[0]} samples from {df.index.min()} to {df.index.max()}")
        print(f"📈 Added {len(self.qlib_alpha_factors)} Qlib Alpha features")
        return df
    
    def add_qlib_alpha_features(self, df):
        """Add Qlib's built-in Alpha features"""
        print("🧮 Adding Qlib Alpha features...")
        
        # Create synthetic stock data for Qlib (since we have single stock data)
        # We'll simulate a multi-stock environment by creating variations
        
        # Alpha158 features (most commonly used)
        alpha158_features = self.calculate_alpha158_features(df)
        
        # Alpha360 features (more comprehensive)
        alpha360_features = self.calculate_alpha360_features(df)
        
        # Combine all features
        for feature_name, feature_values in alpha158_features.items():
            df[f'alpha158_{feature_name}'] = feature_values
            self.qlib_alpha_factors[f'alpha158_{feature_name}'] = feature_values
            
        for feature_name, feature_values in alpha360_features.items():
            df[f'alpha360_{feature_name}'] = feature_values
            self.qlib_alpha_factors[f'alpha360_{feature_name}'] = feature_values
        
        return df
    
    def calculate_alpha158_features(self, df):
        """Calculate Alpha158 features"""
        features = {}
        
        # Price-based features
        features['price_momentum'] = df['close'].pct_change(5)
        features['price_reversal'] = -df['close'].pct_change(1)
        features['price_acceleration'] = df['close'].pct_change(5).diff()
        
        # Volume-based features
        features['volume_momentum'] = df['volume'].pct_change(5)
        features['volume_price_trend'] = df['volume'].pct_change() * df['close'].pct_change()
        
        # Safe division for volume ratio
        volume_ma = df['volume'].rolling(20).mean()
        features['volume_ma_ratio'] = df['volume'] / volume_ma.replace(0, np.nan)
        
        # Volatility features
        features['volatility'] = df['returns'].rolling(20).std()
        
        # Safe division for volatility ratio
        vol_5 = df['returns'].rolling(5).std()
        vol_20 = df['returns'].rolling(20).std()
        features['volatility_ratio'] = vol_5 / vol_20.replace(0, np.nan)
        
        # Technical indicators
        features['rsi'] = self.calculate_rsi(df['close'])
        features['macd'] = self.calculate_macd(df['close'])
        features['bollinger_position'] = self.calculate_bollinger_position(df['close'])
        
        # Cross-sectional features (simulated)
        features['cross_sectional_momentum'] = df['returns'].rolling(10).rank(pct=True)
        features['cross_sectional_reversal'] = 1 - df['returns'].rolling(10).rank(pct=True)
        
        return features
    
    def calculate_alpha360_features(self, df):
        """Calculate Alpha360 features (more comprehensive)"""
        features = {}
        
        # Enhanced price features
        features['price_momentum_short'] = df['close'].pct_change(1)
        features['price_momentum_medium'] = df['close'].pct_change(5)
        features['price_momentum_long'] = df['close'].pct_change(20)
        
        # Enhanced volume features
        features['volume_force'] = df['volume'] * df['returns']
        features['volume_price_correlation'] = df['returns'].rolling(10).corr(df['volume'].pct_change())
        
        # Enhanced volatility features
        features['realized_volatility'] = df['returns'].rolling(20).std() * np.sqrt(252 * 390)
        features['volatility_of_volatility'] = features['realized_volatility'].rolling(10).std()
        
        # Clean any infinite values in these features
        features['realized_volatility'] = features['realized_volatility'].replace([np.inf, -np.inf], np.nan)
        features['volatility_of_volatility'] = features['volatility_of_volatility'].replace([np.inf, -np.inf], np.nan)
        
        # Microstructure features
        # Safe division for bid-ask spread
        features['bid_ask_spread'] = (df['high'] - df['low']) / df['close'].replace(0, np.nan)  # Proxy for spread
        
        # Advanced price efficiency (manual autocorrelation calculation)
        def rolling_autocorr(x, lag=1):
            if len(x) < lag + 1:
                return np.nan
            return np.corrcoef(x[:-lag], x[lag:])[0, 1]
        
        features['price_efficiency'] = 1 - df['returns'].rolling(20).apply(lambda x: rolling_autocorr(x, 1), raw=True)
        
        # Enhanced technical features
        features['stochastic'] = self.calculate_stochastic(df)
        features['williams_r'] = self.calculate_williams_r(df)
        features['cci'] = self.calculate_cci(df)
        
        return features
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd - signal_line
    
    def calculate_bollinger_position(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Band position"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return (prices - lower) / (upper - lower)
    
    def calculate_stochastic(self, df, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        lowest_low = df['low'].rolling(k_period).min()
        highest_high = df['high'].rolling(k_period).max()
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        return k_percent.rolling(d_period).mean()
    
    def calculate_williams_r(self, df, period=14):
        """Calculate Williams %R"""
        highest_high = df['high'].rolling(period).max()
        lowest_low = df['low'].rolling(period).min()
        return -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
    
    def calculate_cci(self, df, period=20):
        """Calculate Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma) / (0.015 * mad)
    
    def calculate_feature_importance(self, df):
        """Calculate feature importance using correlation with future returns"""
        future_returns = df['returns'].shift(-1)
        correlations = {}
        
        for col in df.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume', 'returns', 'log_returns']:
                corr = df[col].corr(future_returns)
                correlations[col] = abs(corr) if not pd.isna(corr) else 0
        
        # Return average importance
        return np.mean(list(correlations.values())) if correlations else 0
    
    def calculate_regime_classification(self, df):
        """Calculate regime classification score"""
        vol_ratio = df['returns'].rolling(5).std() / df['returns'].rolling(20).std()
        momentum = df['close'].pct_change(10)
        volume_ratio = df['volume'] / df['volume'].rolling(20).mean()
        
        # Combine into regime score
        regime_score = (vol_ratio + abs(momentum) + volume_ratio) / 3
        return regime_score.rolling(10).mean()
    
    def compute_alpha_factors(self):
        """Compute sophisticated alpha factors including Qlib features"""
        print("🧮 Computing Alpha Factors with Qlib features...")
        
        df = self.data.copy()
        
        # === CUSTOM MOMENTUM FACTORS ===
        # Multi-timeframe momentum
        df['momentum_5min'] = df['close'].pct_change(5)
        df['momentum_15min'] = df['close'].pct_change(15)
        df['momentum_30min'] = df['close'].pct_change(30)
        df['momentum_60min'] = df['close'].pct_change(60)
        
        # Momentum strength
        df['momentum_strength'] = (df['momentum_5min'] * 0.4 + 
                                 df['momentum_15min'] * 0.3 + 
                                 df['momentum_30min'] * 0.2 + 
                                 df['momentum_60min'] * 0.1)
        
        # === MEAN REVERSION FACTORS ===
        # Distance from moving averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_30'] = df['close'].rolling(30).mean()
        df['sma_60'] = df['close'].rolling(60).mean()
        
        df['mean_reversion_10'] = (df['close'] - df['sma_10']) / df['sma_10']
        df['mean_reversion_30'] = (df['close'] - df['sma_30']) / df['sma_30']
        df['mean_reversion_60'] = (df['close'] - df['sma_60']) / df['sma_60']
        
        # === VOLATILITY FACTORS ===
        # Realized volatility (multiple windows)
        df['vol_5min'] = df['returns'].rolling(5).std() * np.sqrt(252 * 390)  # Annualized
        df['vol_15min'] = df['returns'].rolling(15).std() * np.sqrt(252 * 390)
        df['vol_30min'] = df['returns'].rolling(30).std() * np.sqrt(252 * 390)
        
        # Volatility ratio (regime detector)
        df['vol_ratio'] = df['vol_5min'] / df['vol_30min']
        
        # === VOLUME FACTORS ===
        # Volume momentum
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_momentum'] = df['volume'] / df['volume_sma']
        
        # Volume-price relationship
        df['volume_price_corr'] = df['returns'].rolling(20).corr(df['volume'].pct_change())
        
        # === MICROSTRUCTURE FACTORS ===
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Price efficiency (random walk test)
        df['price_efficiency'] = df['returns'].rolling(20).apply(
            lambda x: 1 - (x.autocorr() if len(x.dropna()) > 1 else 0)
        )
        
        # === TECHNICAL FACTORS ===
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_momentum'] = df['rsi'].diff()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # === REGIME FACTORS ===
        # Market stress indicator
        df['stress_indicator'] = (df['vol_ratio'] * abs(df['mean_reversion_30']) * 
                                df['hl_spread']).rolling(10).mean()
        
        # Trend strength
        df['trend_strength'] = abs(df['momentum_30min']) / df['vol_30min']
        
        self.data = df
        
        # Store alpha factors for analysis (including Qlib features)
        self.alpha_factors = {
            'momentum_strength': df['momentum_strength'],
            'mean_reversion_30': df['mean_reversion_30'],
            'vol_ratio': df['vol_ratio'],
            'volume_momentum': df['volume_momentum'],
            'price_efficiency': df['price_efficiency'],
            'rsi_momentum': df['rsi_momentum'],
            'bb_position': df['bb_position'],
            'macd_histogram': df['macd_histogram'],
            'stress_indicator': df['stress_indicator'],
            'trend_strength': df['trend_strength']
        }
        
        # Add Qlib features to alpha factors
        for name, series in self.qlib_alpha_factors.items():
            if not series.empty:
                self.alpha_factors[name] = series
        
        print(f"✅ Computed {len(self.alpha_factors)} alpha factors (including {len(self.qlib_alpha_factors)} Qlib features)")
        return df
    
    def detect_market_regimes(self):
        """Detect market regimes using machine learning"""
        print("🎯 Detecting Market Regimes...")
        
        df = self.data.copy()
        
        # Features for regime detection (including Qlib features)
        features = ['vol_ratio', 'trend_strength', 'stress_indicator', 
                   'volume_momentum', 'price_efficiency']
        
        # Add top Qlib features
        qlib_features = [col for col in df.columns if col.startswith(('alpha158_', 'alpha360_'))]
        features.extend(qlib_features[:5])  # Top 5 Qlib features
        
        regime_data = df[features].dropna()
        
        # Clean infinite and extreme values
        regime_data = regime_data.replace([np.inf, -np.inf], np.nan)
        regime_data = regime_data.dropna()
        
        # Handle extreme values by clipping
        for col in regime_data.columns:
            Q1 = regime_data[col].quantile(0.01)
            Q3 = regime_data[col].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            regime_data[col] = regime_data[col].clip(lower_bound, upper_bound)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(regime_data)
        
        # Use K-means to identify regimes
        n_regimes = 4  # Normal, High Vol, Trending, Stressed
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regimes = kmeans.fit_predict(scaled_features)
        
        # Map regimes to interpretable names
        regime_names = {0: 'Normal', 1: 'High_Volatility', 2: 'Trending', 3: 'Stressed'}
        
        df.loc[regime_data.index, 'regime'] = regimes
        df['regime_name'] = df['regime'].map(regime_names)
        
        # Regime persistence
        df['regime_duration'] = df.groupby((df['regime'] != df['regime'].shift()).cumsum()).cumcount() + 1
        
        self.data = df
        self.regime_states = regime_names
        
        print(f"✅ Identified market regimes: {list(regime_names.values())}")
        return df
    
    def compute_risk_metrics(self):
        """Compute comprehensive risk metrics"""
        print("⚖️ Computing Risk Metrics...")
        
        df = self.data.copy()
        returns = df['returns'].dropna()
        
        # Basic risk metrics
        volatility = returns.std() * np.sqrt(252 * 390)  # Annualized
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (assuming risk-free rate = 0)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 390)
        
        # Calmar Ratio
        annual_return = (1 + returns.mean()) ** (252 * 390) - 1
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = returns.mean() / downside_std * np.sqrt(252 * 390) if downside_std != 0 else 0
        
        self.risk_metrics = {
            'volatility': volatility,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio
        }
        
        print("✅ Risk metrics computed")
        return self.risk_metrics
    
    def create_composite_alpha_signal(self):
        """Create composite alpha signal using factor scoring with Qlib features"""
        print("🔮 Creating Composite Alpha Signal with Qlib features...")
        
        df = self.data.copy()
        
        # Enhanced regime-dependent factor weights (including Qlib features)
        regime_weights = {
            'Normal': {
                'momentum_strength': 0.20,
                'mean_reversion_30': 0.25,
                'vol_ratio': -0.05,
                'rsi_momentum': 0.15,
                'macd_histogram': 0.15,
                'bb_position': 0.10,
                'alpha158_price_momentum': 0.10
            },
            'High_Volatility': {
                'mean_reversion_30': 0.30,
                'vol_ratio': -0.25,
                'stress_indicator': -0.20,
                'bb_position': 0.15,
                'alpha158_volatility': 0.20
            },
            'Trending': {
                'momentum_strength': 0.35,
                'trend_strength': 0.25,
                'macd_histogram': 0.20,
                'alpha158_price_momentum': 0.20
            },
            'Stressed': {
                'mean_reversion_30': 0.45,
                'stress_indicator': -0.35,
                'alpha158_volatility': 0.20
            }
        }
        
        # Initialize alpha signal
        df['alpha_signal'] = 0
        
        # Apply regime-specific weights
        for regime_name, weights in regime_weights.items():
            mask = df['regime_name'] == regime_name
            signal = 0
            
            for factor, weight in weights.items():
                if factor in df.columns:
                    # Normalize factor to [-1, 1] range
                    factor_norm = df[factor].rolling(60).rank(pct=True) * 2 - 1
                    signal += weight * factor_norm
            
            df.loc[mask, 'alpha_signal'] = signal[mask]
        
        # Signal smoothing
        df['alpha_signal_smooth'] = df['alpha_signal'].rolling(5).mean()
        
        # Signal confidence based on regime stability and Qlib feature quality
        df['signal_confidence'] = 1 / (1 + df['regime_duration'].rolling(10).std())
        
        # Enhance confidence with Qlib feature quality
        qlib_quality = df[[col for col in df.columns if col.startswith(('alpha158_', 'alpha360_', 'alpha191_'))]].abs().mean(axis=1)
        df['signal_confidence'] = (df['signal_confidence'] + qlib_quality.rolling(20).mean()) / 2
        
        # Final signal with confidence adjustment
        df['final_signal'] = df['alpha_signal_smooth'] * df['signal_confidence']
        
        # Add minimum signal strength filter
        min_signal_strength = 0.15  # Only trade if signal is strong enough
        df['final_signal'] = df['final_signal'].apply(lambda x: x if abs(x) > min_signal_strength else 0)
        
        # Position sizing based on volatility
        df['position_size'] = 1 / (1 + df['vol_5min'] * 10)  # Inverse volatility weighting
        
        self.data = df
        print("✅ Composite alpha signal created with Qlib features")
        return df
    
    def portfolio_optimization(self):
        """Advanced portfolio optimization with position sizing, SL, and TP"""
        print("📈 Optimizing Portfolio with SL/TP...")
        
        df = self.data.copy()
        
        # Risk budgeting approach
        target_vol = 0.15  # 15% annual volatility target
        current_vol = df['vol_30min'].iloc[-1] if not pd.isna(df['vol_30min'].iloc[-1]) else 0.2
        
        # Kelly Criterion for position sizing
        signal = df['final_signal'].iloc[-1] if not pd.isna(df['final_signal'].iloc[-1]) else 0
        win_rate = (df['returns'][df['final_signal'].shift(1) > 0] > 0).mean()
        avg_win = df['returns'][df['returns'] > 0].mean()
        avg_loss = abs(df['returns'][df['returns'] < 0].mean())
        
        if avg_loss != 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        else:
            kelly_fraction = 0.1
        
        # Volatility scaling
        vol_scaling = target_vol / current_vol if current_vol > 0 else 1
        vol_scaling = max(0.1, min(vol_scaling, 2.0))  # Reasonable bounds
        
        # Final position size
        base_position = signal * kelly_fraction * vol_scaling
        
        # Risk management overlay
        if abs(signal) < 0.15:  # Weak signal (increased threshold)
            base_position *= 0.3  # More aggressive reduction
        
        if df['stress_indicator'].iloc[-1] > df['stress_indicator'].quantile(0.9):  # High stress
            base_position *= 0.2  # More aggressive reduction
        
        # Regime-based position sizing
        regime = df['regime_name'].iloc[-1]
        if regime == 'Stressed':
            base_position *= 0.1  # Very small positions in stressed markets
        elif regime == 'High_Volatility':
            base_position *= 0.5  # Reduced positions in high volatility
        elif regime == 'Normal':
            base_position *= 0.8  # Normal positions
        # Trending regime keeps full position
        
        # Portfolio weights (use optimized position cap if available)
        if hasattr(self, 'optimization_params'):
            pos_cap = self.optimization_params['position_size_cap']
        else:
            pos_cap = 0.5
        optimal_weight = max(-pos_cap, min(pos_cap, base_position))
        
        # Calculate Stop Loss and Take Profit levels
        current_price = df['close'].iloc[-1]
        current_volatility = df['vol_5min'].iloc[-1] if not pd.isna(df['vol_5min'].iloc[-1]) else 0.2
        
        # Dynamic SL/TP based on volatility and regime
        regime = df['regime_name'].iloc[-1]
        
        # Use optimized parameters if available, otherwise use regime-based defaults
        if hasattr(self, 'optimization_params'):
            sl_multiplier = self.optimization_params['sl_multiplier']
            tp_multiplier = self.optimization_params['tp_multiplier']
        else:
            # Base SL/TP multipliers (in standard deviations)
            if regime == 'Normal':
                sl_multiplier = 2.0  # 2 std dev for SL
                tp_multiplier = 3.0  # 3 std dev for TP
            elif regime == 'High_Volatility':
                sl_multiplier = 1.5  # Tighter SL in high vol
                tp_multiplier = 2.5  # Lower TP in high vol
            elif regime == 'Trending':
                sl_multiplier = 2.5  # Wider SL in trending
                tp_multiplier = 4.0  # Higher TP in trending
            else:  # Stressed
                sl_multiplier = 1.0  # Very tight SL
                tp_multiplier = 2.0  # Lower TP
        
        # Convert volatility to price levels
        price_volatility = current_price * current_volatility / np.sqrt(252 * 390)
        
        # Calculate SL and TP levels (more conservative)
        if signal > 0:  # BUY signal
            stop_loss = current_price - (sl_multiplier * price_volatility * 0.8)  # Tighter stop loss
            take_profit = current_price + (tp_multiplier * price_volatility * 1.2)  # Higher target
            position_type = "LONG"
        elif signal < 0:  # SELL signal
            stop_loss = current_price + (sl_multiplier * price_volatility * 0.8)  # Tighter stop loss
            take_profit = current_price - (tp_multiplier * price_volatility * 1.2)  # Higher target
            position_type = "SHORT"
        else:  # No signal
            stop_loss = take_profit = current_price
            position_type = "FLAT"
        
        # Risk per trade (use optimized if available)
        if hasattr(self, 'optimization_params'):
            risk_per_trade = self.optimization_params['risk_per_trade']
        else:
            risk_per_trade = 0.005  # More conservative risk per trade
        account_size = 100000  # Assume ₹1L account
        
        # Position size calculation based on risk
        if position_type != "FLAT":
            risk_amount = account_size * risk_per_trade
            price_risk = abs(current_price - stop_loss)
            position_size = risk_amount / price_risk
            
            # Cap position size between 1 and 10 for testing
            position_size = max(1, min(position_size, 10))
            
            position_value = position_size * current_price
            position_percentage = (position_value / account_size) * 100
        else:
            position_size = position_value = position_percentage = 0
        
        # Risk metrics for the position
        risk_reward_ratio = abs(take_profit - current_price) / abs(stop_loss - current_price) if position_type != "FLAT" else 0
        max_loss = abs(current_price - stop_loss) * position_size if position_type != "FLAT" else 0
        max_profit = abs(take_profit - current_price) * position_size if position_type != "FLAT" else 0
        
        self.portfolio_weights = {
            'CESC': optimal_weight,
            'kelly_fraction': kelly_fraction,
            'vol_scaling': vol_scaling,
            'signal_strength': abs(signal),
            'regime': regime,
            'position_type': position_type,
            'position_size': position_size,
            'position_value': position_value,
            'position_percentage': position_percentage,
            'current_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward_ratio,
            'max_loss': max_loss,
            'max_profit': max_profit,
            'sl_distance_bps': abs(stop_loss - current_price) / current_price * 10000,
            'tp_distance_bps': abs(take_profit - current_price) / current_price * 10000
        }
        
        print(f"✅ Portfolio optimized - {position_type} Position")
        print(f"   Size: {position_size:.0f} shares (₹{position_value:.0f})")
        print(f"   SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f}")
        print(f"   R:R Ratio: {risk_reward_ratio:.2f}")
        
        return self.portfolio_weights
    
    def backtest_strategy(self):
        """Comprehensive backtesting with transaction costs and comparison"""
        print("📊 Running Strategy Backtest with Buy & Hold comparison...")
        
        df = self.data.copy()
        
        # Signal with lag (realistic trading)
        df['signal_lagged'] = df['final_signal'].shift(1)
        
        # Transaction costs (0.1% per trade)
        transaction_cost = 0.001
        df['position_change'] = df['signal_lagged'].diff().abs()
        df['transaction_costs'] = df['position_change'] * transaction_cost
        
        # Strategy returns
        df['strategy_returns'] = df['signal_lagged'] * df['returns'] - df['transaction_costs']
        
        # Buy & Hold returns
        df['buy_hold_returns'] = df['returns']
        
        # Performance metrics for strategy
        total_return = (1 + df['strategy_returns']).prod() - 1
        annual_return = (1 + df['strategy_returns'].mean()) ** (252 * 390) - 1
        annual_vol = df['strategy_returns'].std() * np.sqrt(252 * 390)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown analysis for strategy
        cumulative = (1 + df['strategy_returns']).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # Hit rate for strategy
        hit_rate = (df['strategy_returns'] > 0).mean()
        
        # Profit factor for strategy
        gross_profit = df['strategy_returns'][df['strategy_returns'] > 0].sum()
        gross_loss = abs(df['strategy_returns'][df['strategy_returns'] < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Buy & Hold performance metrics
        bh_total_return = (1 + df['buy_hold_returns']).prod() - 1
        bh_annual_return = (1 + df['buy_hold_returns'].mean()) ** (252 * 390) - 1
        bh_annual_vol = df['buy_hold_returns'].std() * np.sqrt(252 * 390)
        bh_sharpe = bh_annual_return / bh_annual_vol if bh_annual_vol > 0 else 0
        
        # Buy & Hold drawdown
        bh_cumulative = (1 + df['buy_hold_returns']).cumprod()
        bh_rolling_max = bh_cumulative.expanding().max()
        bh_drawdown = (bh_cumulative - bh_rolling_max) / bh_rolling_max
        bh_max_dd = bh_drawdown.min()
        
        backtest_results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'hit_rate': hit_rate,
            'profit_factor': profit_factor,
            'num_trades': df['position_change'].sum() / 2,  # Round trips
            'avg_trade_duration': df.groupby((df['signal_lagged'] != df['signal_lagged'].shift()).cumsum()).size().mean()
        }
        
        # Comparison metrics
        self.comparison_metrics = {
            'strategy_vs_bh': {
                'excess_return': total_return - bh_total_return,
                'excess_annual_return': annual_return - bh_annual_return,
                'excess_sharpe': sharpe - bh_sharpe,
                'excess_max_dd': max_dd - bh_max_dd,
                'information_ratio': (annual_return - bh_annual_return) / (annual_vol - bh_annual_vol) if (annual_vol - bh_annual_vol) != 0 else 0
            },
            'buy_hold': {
                'total_return': bh_total_return,
                'annual_return': bh_annual_return,
                'annual_volatility': bh_annual_vol,
                'sharpe_ratio': bh_sharpe,
                'max_drawdown': bh_max_dd
            }
        }
        
        self.backtest_results = backtest_results
        self.data = df
        
        print("✅ Backtest completed with Buy & Hold comparison")
        print(f"📈 Strategy vs Buy & Hold:")
        print(f"   Excess Return: {self.comparison_metrics['strategy_vs_bh']['excess_return']:.2%}")
        print(f"   Excess Sharpe: {self.comparison_metrics['strategy_vs_bh']['excess_sharpe']:.2f}")
        print(f"   Information Ratio: {self.comparison_metrics['strategy_vs_bh']['information_ratio']:.2f}")
        
        return backtest_results
    
    def generate_live_trading_signal(self):
        """Generate current trading signal for live implementation"""
        print("🔴 LIVE TRADING SIGNAL")
        print("=" * 60)
        
        latest = self.data.iloc[-1]
        
        signal_strength = abs(latest['final_signal'])
        direction = "BUY" if latest['final_signal'] > 0 else "SELL" if latest['final_signal'] < 0 else "HOLD"
        confidence = latest['signal_confidence']
        regime = latest['regime_name']
        
        # Get position details
        pos_type = self.portfolio_weights['position_type']
        pos_size = self.portfolio_weights['position_size']
        pos_value = self.portfolio_weights['position_value']
        current_price = self.portfolio_weights['current_price']
        stop_loss = self.portfolio_weights['stop_loss']
        take_profit = self.portfolio_weights['take_profit']
        rr_ratio = self.portfolio_weights['risk_reward_ratio']
        max_loss = self.portfolio_weights['max_loss']
        max_profit = self.portfolio_weights['max_profit']
        
        print(f"🎯 SIGNAL: {direction}")
        print(f"💪 Strength: {signal_strength:.3f}")
        print(f"✅ Confidence: {confidence:.3f}")
        print(f"🏛️ Regime: {regime}")
        print(f"📊 Current Price: ₹{current_price:.2f}")
        print(f"📈 Expected Move: {latest['final_signal'] * 100:.2f} bps")
        print()
        print("💰 POSITION DETAILS:")
        print(f"   Type: {pos_type}")
        print(f"   Size: {pos_size:.0f} shares")
        print(f"   Value: ₹{pos_value:.0f}")
        print(f"   % of Account: {self.portfolio_weights['position_percentage']:.1f}%")
        print()
        print("🎯 RISK MANAGEMENT:")
        print(f"   Stop Loss: ₹{stop_loss:.2f} ({self.portfolio_weights['sl_distance_bps']:.0f} bps)")
        print(f"   Take Profit: ₹{take_profit:.2f} ({self.portfolio_weights['tp_distance_bps']:.0f} bps)")
        print(f"   Risk:Reward Ratio: {rr_ratio:.2f}")
        print(f"   Max Loss: ₹{max_loss:.0f}")
        print(f"   Max Profit: ₹{max_profit:.0f}")
        print()
        print("⚡ EXECUTION SUMMARY:")
        if pos_type == "LONG":
            print(f"   BUY {pos_size:.0f} shares at ₹{current_price:.2f}")
            print(f"   SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f}")
        elif pos_type == "SHORT":
            print(f"   SELL {pos_size:.0f} shares at ₹{current_price:.2f}")
            print(f"   SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f}")
        else:
            print("   NO POSITION - Stay flat")
        
        return {
            'signal': direction,
            'strength': signal_strength,
            'confidence': confidence,
            'position_type': pos_type,
            'position_size': pos_size,
            'position_value': pos_value,
            'current_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': rr_ratio,
            'max_loss': max_loss,
            'max_profit': max_profit,
            'regime': regime,
            'timestamp': latest.name
        }
    
    def get_dashboard_data(self):
        """Get all dashboard data as JSON"""
        df = self.data.dropna()
        
        # Prepare data for web interface
        dashboard_data = {
            'price_data': {
                'timestamps': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'prices': df['close'].tolist(),
                'signals': df['final_signal'].tolist(),
                'volumes': df['volume'].tolist()
            },
            'performance': {
                'total_return': self.backtest_results['total_return'],
                'annual_return': self.backtest_results['annual_return'],
                'sharpe_ratio': self.backtest_results['sharpe_ratio'],
                'max_drawdown': self.backtest_results['max_drawdown'],
                'hit_rate': self.backtest_results['hit_rate'],
                'profit_factor': self.backtest_results['profit_factor']
            },
            'current_signal': {
                'direction': "BUY" if self.data['final_signal'].iloc[-1] > 0 else "SELL" if self.data['final_signal'].iloc[-1] < 0 else "HOLD",
                'strength': abs(self.data['final_signal'].iloc[-1]),
                'confidence': self.data['signal_confidence'].iloc[-1],
                'regime': self.data['regime_name'].iloc[-1],
                'current_price': self.data['close'].iloc[-1]
            },
            'position': self.portfolio_weights,
            'risk_metrics': self.risk_metrics,
            'regime_distribution': self.data['regime_name'].value_counts().to_dict(),
            'alpha_factors': {name: series.iloc[-1] if not series.empty else 0 
                            for name, series in self.alpha_factors.items()},
            'comparison_metrics': self.comparison_metrics,
            'qlib_features_count': len(self.qlib_alpha_factors)
        }
        
        return dashboard_data

    def auto_optimize_parameters(self, target_return=0.20, target_sharpe=1.0, target_max_dd=-0.10, param_bounds=None):
        """
        Auto-optimize parameters to achieve target performance metrics
        
        Args:
            target_return: Target annual return (e.g., 0.20 = 20%)
            target_sharpe: Target Sharpe ratio (e.g., 1.0)
            target_max_dd: Target maximum drawdown (e.g., -0.10 = -10%)
        """
        print("🔧 Starting Auto-Optimization...")
        print(f"🎯 Targets: Return={target_return*100:.1f}%, Sharpe={target_sharpe:.2f}, MaxDD={target_max_dd*100:.1f}%")
        
        from scipy.optimize import minimize
        import numpy as np
        
        # Use custom parameter bounds if provided, otherwise use defaults
        if param_bounds:
            param_bounds_list = [
                (param_bounds.get('sl_multiplier', {}).get('min', 0.5), param_bounds.get('sl_multiplier', {}).get('max', 3.0)),
                (param_bounds.get('tp_multiplier', {}).get('min', 1.0), param_bounds.get('tp_multiplier', {}).get('max', 5.0)),
                (param_bounds.get('risk_per_trade', {}).get('min', 0.001), param_bounds.get('risk_per_trade', {}).get('max', 0.02)),
                (param_bounds.get('min_signal_strength', {}).get('min', 0.05), param_bounds.get('min_signal_strength', {}).get('max', 0.3)),
                (param_bounds.get('position_size_cap', {}).get('min', 0.1), param_bounds.get('position_size_cap', {}).get('max', 0.8)),
            ]
        else:
            param_bounds_list = [
                (0.5, 3.0),    # SL multiplier
                (1.0, 5.0),    # TP multiplier
                (0.001, 0.02), # Risk per trade
                (0.05, 0.3),   # Min signal strength (more conservative)
                (0.1, 0.8),    # Position size cap (more conservative)
            ]
        
        def objective_function(params):
            """Objective function to minimize (distance from targets)"""
            try:
                # Extract parameters
                sl_mult, tp_mult, risk_per_trade, min_signal, pos_cap = params
                
                # Store original state
                original_data = self.data.copy()
                original_optimization_params = getattr(self, 'optimization_params', None)
                
                # Apply parameter changes temporarily
                self._apply_optimization_params(sl_mult, tp_mult, risk_per_trade, min_signal, pos_cap)
                
                # Re-run backtest with new parameters
                self.backtest_strategy()
                
                # Get performance metrics
                results = self.backtest_results
                actual_return = results['annual_return']
                actual_sharpe = results['sharpe_ratio']
                actual_max_dd = results['max_drawdown']
                
                # Calculate distance from targets with better normalization
                return_error = abs(actual_return - target_return) / max(abs(target_return), 0.01)
                sharpe_error = abs(actual_sharpe - target_sharpe) / max(abs(target_sharpe), 0.1)
                dd_error = abs(actual_max_dd - target_max_dd) / max(abs(target_max_dd), 0.01)
                
                # Combined error (weighted)
                total_error = (return_error * 0.4 + sharpe_error * 0.4 + dd_error * 0.2)
                
                # Add penalty for very poor performance
                if actual_return < -0.5:  # -50% return
                    total_error += 10
                
                # Restore original state
                self.data = original_data
                if original_optimization_params is None:
                    if hasattr(self, 'optimization_params'):
                        delattr(self, 'optimization_params')
                else:
                    self.optimization_params = original_optimization_params
                
                return total_error
                
            except Exception as e:
                print(f"⚠️ Optimization error: {e}")
                return 1000  # High penalty for errors
        
        # Run optimization
        print("🔄 Running parameter optimization...")
        result = minimize(
            objective_function,
            x0=[1.5, 2.5, 0.005, 0.1, 0.3],  # More conservative initial guess
            bounds=param_bounds_list,
            method='L-BFGS-B',
            options={'maxiter': 30, 'ftol': 1e-4}
        )
        
        if result.success:
            optimal_params = result.x
            sl_mult, tp_mult, risk_per_trade, min_signal, pos_cap = optimal_params
            
            print("✅ Optimization completed!")
            print(f"📊 Optimal Parameters:")
            print(f"   SL Multiplier: {sl_mult:.2f}")
            print(f"   TP Multiplier: {tp_mult:.2f}")
            print(f"   Risk per Trade: {risk_per_trade:.3f}")
            print(f"   Min Signal Strength: {min_signal:.2f}")
            print(f"   Position Size Cap: {pos_cap:.2f}")
            
            # Apply optimal parameters
            self._apply_optimization_params(sl_mult, tp_mult, risk_per_trade, min_signal, pos_cap)
            self.portfolio_optimization()
            self.backtest_strategy()
            
            # Check if results are acceptable
            if self.backtest_results['annual_return'] < -0.5:  # -50% return
                print("⚠️ Optimization resulted in very poor performance. Keeping original parameters.")
                return {
                    'success': False,
                    'error': 'Optimization resulted in very poor performance (-50%+ loss). Consider adjusting target metrics or parameter bounds.'
                }
            
            return {
                'success': True,
                'optimal_params': {
                    'sl_multiplier': sl_mult,
                    'tp_multiplier': tp_mult,
                    'risk_per_trade': risk_per_trade,
                    'min_signal_strength': min_signal,
                    'position_size_cap': pos_cap
                },
                'performance': self.backtest_results
            }
        else:
            print("❌ Optimization failed")
            return {'success': False, 'error': 'Optimization did not converge'}
    
    def _apply_optimization_params(self, sl_mult, tp_mult, risk_per_trade, min_signal, pos_cap):
        """Apply optimization parameters to the engine"""
        # Store parameters for use in portfolio optimization
        self.optimization_params = {
            'sl_multiplier': sl_mult,
            'tp_multiplier': tp_mult,
            'risk_per_trade': risk_per_trade,
            'min_signal_strength': min_signal,
            'position_size_cap': pos_cap
        }
        
        # Update signal strength filter
        df = self.data.copy()
        df['final_signal'] = df['final_signal'].apply(lambda x: x if abs(x) > min_signal else 0)
        self.data = df
