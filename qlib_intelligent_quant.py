#!/usr/bin/env python3
"""
Intelligent Quantitative Trading System using Microsoft Qlib
Author: AI Assistant
Description: Advanced quantitative analysis using multiple alpha factors, risk models,
            portfolio optimization, and regime detection for live trading readiness.
            Now with FastAPI web interface.
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

# Web framework
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import json
from datetime import datetime
import os

# Plotting
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils

class QuantitativeAlphaEngine:
    """
    Advanced Quantitative Alpha Engine using Qlib's capabilities
    """
    
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = None
        self.alpha_factors = {}
        self.regime_states = None
        self.risk_metrics = {}
        self.portfolio_weights = None
        
    def load_and_prepare_data(self):
        """Load and prepare 1-minute data with advanced preprocessing"""
        print("📊 Loading and preparing 1-minute data...")
        
        # Load data
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
        
        # Remove NaN values
        df.dropna(inplace=True)
        
        self.data = df
        print(f"✅ Data prepared: {df.shape[0]} samples from {df.index.min()} to {df.index.max()}")
        return df
    
    def compute_alpha_factors(self):
        """Compute sophisticated alpha factors"""
        print("🧮 Computing Alpha Factors...")
        
        df = self.data.copy()
        
        # === MOMENTUM FACTORS ===
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
        
        # Store alpha factors for analysis
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
        
        print(f"✅ Computed {len(self.alpha_factors)} alpha factors")
        return df
    
    def detect_market_regimes(self):
        """Detect market regimes using machine learning"""
        print("🎯 Detecting Market Regimes...")
        
        df = self.data.copy()
        
        # Features for regime detection
        features = ['vol_ratio', 'trend_strength', 'stress_indicator', 
                   'volume_momentum', 'price_efficiency']
        
        regime_data = df[features].dropna()
        
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
        """Create composite alpha signal using factor scoring"""
        print("🔮 Creating Composite Alpha Signal...")
        
        df = self.data.copy()
        
        # Regime-dependent factor weights
        regime_weights = {
            'Normal': {
                'momentum_strength': 0.2,
                'mean_reversion_30': 0.3,
                'vol_ratio': -0.1,
                'rsi_momentum': 0.2,
                'macd_histogram': 0.2,
                'bb_position': 0.1
            },
            'High_Volatility': {
                'mean_reversion_30': 0.4,
                'vol_ratio': -0.3,
                'stress_indicator': -0.2,
                'bb_position': 0.1
            },
            'Trending': {
                'momentum_strength': 0.5,
                'trend_strength': 0.3,
                'macd_histogram': 0.2
            },
            'Stressed': {
                'mean_reversion_30': 0.6,
                'stress_indicator': -0.4
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
        
        # Signal confidence based on regime stability
        df['signal_confidence'] = 1 / (1 + df['regime_duration'].rolling(10).std())
        
        # Final signal with confidence adjustment
        df['final_signal'] = df['alpha_signal_smooth'] * df['signal_confidence']
        
        # Position sizing based on volatility
        df['position_size'] = 1 / (1 + df['vol_5min'] * 10)  # Inverse volatility weighting
        
        self.data = df
        print("✅ Composite alpha signal created")
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
        if abs(signal) < 0.1:  # Weak signal
            base_position *= 0.5
        
        if df['stress_indicator'].iloc[-1] > df['stress_indicator'].quantile(0.9):  # High stress
            base_position *= 0.3
        
        # Portfolio weights (single asset for this example)
        optimal_weight = max(-1, min(1, base_position))  # Constrain to [-1, 1]
        
        # Calculate Stop Loss and Take Profit levels
        current_price = df['close'].iloc[-1]
        current_volatility = df['vol_5min'].iloc[-1] if not pd.isna(df['vol_5min'].iloc[-1]) else 0.2
        
        # Dynamic SL/TP based on volatility and regime
        regime = df['regime_name'].iloc[-1]
        
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
        
        # Calculate SL and TP levels
        if signal > 0:  # BUY signal
            stop_loss = current_price - (sl_multiplier * price_volatility)
            take_profit = current_price + (tp_multiplier * price_volatility)
            position_type = "LONG"
        elif signal < 0:  # SELL signal
            stop_loss = current_price + (sl_multiplier * price_volatility)
            take_profit = current_price - (tp_multiplier * price_volatility)
            position_type = "SHORT"
        else:  # No signal
            stop_loss = take_profit = current_price
            position_type = "FLAT"
        
        # Risk per trade (1% of capital)
        risk_per_trade = 0.01
        account_size = 100000  # Assume ₹1L account
        
        # Position size calculation based on risk
        if position_type != "FLAT":
            risk_amount = account_size * risk_per_trade
            price_risk = abs(current_price - stop_loss)
            position_size = risk_amount / price_risk
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
        """Comprehensive backtesting with transaction costs"""
        print("📊 Running Strategy Backtest...")
        
        df = self.data.copy()
        
        # Signal with lag (realistic trading)
        df['signal_lagged'] = df['final_signal'].shift(1)
        
        # Transaction costs (0.1% per trade)
        transaction_cost = 0.001
        df['position_change'] = df['signal_lagged'].diff().abs()
        df['transaction_costs'] = df['position_change'] * transaction_cost
        
        # Strategy returns
        df['strategy_returns'] = df['signal_lagged'] * df['returns'] - df['transaction_costs']
        
        # Performance metrics
        total_return = (1 + df['strategy_returns']).prod() - 1
        annual_return = (1 + df['strategy_returns'].mean()) ** (252 * 390) - 1
        annual_vol = df['strategy_returns'].std() * np.sqrt(252 * 390)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + df['strategy_returns']).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # Hit rate
        hit_rate = (df['strategy_returns'] > 0).mean()
        
        # Profit factor
        gross_profit = df['strategy_returns'][df['strategy_returns'] > 0].sum()
        gross_loss = abs(df['strategy_returns'][df['strategy_returns'] < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
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
        
        self.backtest_results = backtest_results
        self.data = df
        
        print("✅ Backtest completed")
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
    
    def create_dashboard(self):
        """Create comprehensive dashboard"""
        print("📊 Creating Dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        df = self.data.dropna()
        
        # 1. Price and Signal
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df.index, df['close'], label='Price', alpha=0.7, linewidth=1)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(df.index, df['final_signal'], label='Signal', color='red', alpha=0.8, linewidth=2)
        ax1.set_title('Price and Alpha Signal', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Market Regimes
        ax2 = fig.add_subplot(gs[1, 0])
        regime_counts = df['regime_name'].value_counts()
        ax2.pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%')
        ax2.set_title('Market Regimes Distribution', fontweight='bold')
        
        # 3. Alpha Factors Heatmap
        ax3 = fig.add_subplot(gs[1, 1])
        factor_corr = pd.DataFrame(self.alpha_factors).corr()
        sns.heatmap(factor_corr, annot=True, cmap='coolwarm', center=0, ax=ax3, cbar_kws={'shrink': 0.8})
        ax3.set_title('Alpha Factors Correlation', fontweight='bold')
        
        # 4. Strategy Performance
        ax4 = fig.add_subplot(gs[1, 2])
        cumulative_returns = (1 + df['strategy_returns']).cumprod()
        cumulative_benchmark = (1 + df['returns']).cumprod()
        ax4.plot(df.index, cumulative_returns, label='Strategy', linewidth=2)
        ax4.plot(df.index, cumulative_benchmark, label='Buy & Hold', alpha=0.7)
        ax4.set_title('Cumulative Returns', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Risk Metrics
        ax5 = fig.add_subplot(gs[2, 0])
        risk_names = list(self.risk_metrics.keys())[:6]
        risk_values = [self.risk_metrics[name] for name in risk_names]
        bars = ax5.bar(range(len(risk_names)), risk_values)
        ax5.set_xticks(range(len(risk_names)))
        ax5.set_xticklabels(risk_names, rotation=45, ha='right')
        ax5.set_title('Risk Metrics', fontweight='bold')
        
        # Color bars based on values
        for i, (bar, val) in enumerate(zip(bars, risk_values)):
            if 'sharpe' in risk_names[i] or 'calmar' in risk_names[i]:
                bar.set_color('green' if val > 1 else 'orange' if val > 0 else 'red')
            else:
                bar.set_color('red' if val < 0 else 'green')
        
        # 6. Volatility Regimes
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(df.index, df['vol_ratio'], alpha=0.7)
        ax6.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax6.set_title('Volatility Regime Indicator', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. Factor Contributions
        ax7 = fig.add_subplot(gs[2, 2])
        latest_factors = {}
        for name, series in self.alpha_factors.items():
            if not series.empty:
                latest_factors[name] = series.iloc[-1]
        
        factor_names = list(latest_factors.keys())[:8]
        factor_values = [latest_factors[name] for name in factor_names]
        
        colors = ['green' if val > 0 else 'red' for val in factor_values]
        bars = ax7.barh(range(len(factor_names)), factor_values, color=colors, alpha=0.7)
        ax7.set_yticks(range(len(factor_names)))
        ax7.set_yticklabels(factor_names)
        ax7.set_title('Current Factor Values', fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # 8. Performance Summary
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        # Performance table
        perf_text = f"""
        📊 STRATEGY PERFORMANCE SUMMARY
        ════════════════════════════════════════════════════════════════════════
        
        🎯 Returns & Risk:
        • Total Return: {self.backtest_results['total_return']:.2%}
        • Annual Return: {self.backtest_results['annual_return']:.2%}
        • Annual Volatility: {self.backtest_results['annual_volatility']:.2%}
        • Sharpe Ratio: {self.backtest_results['sharpe_ratio']:.2f}
        • Maximum Drawdown: {self.backtest_results['max_drawdown']:.2%}
        
        📈 Trading Statistics:
        • Hit Rate: {self.backtest_results['hit_rate']:.2%}
        • Profit Factor: {self.backtest_results['profit_factor']:.2f}
        • Number of Trades: {self.backtest_results['num_trades']:.0f}
        • Avg Trade Duration: {self.backtest_results['avg_trade_duration']:.1f} minutes
        
        ⚖️ Current Portfolio:
        • Signal: {self.portfolio_weights['CESC']:.3f}
        • Kelly Fraction: {self.portfolio_weights['kelly_fraction']:.3f}
        • Volatility Scaling: {self.portfolio_weights['vol_scaling']:.3f}
        • Market Regime: {self.portfolio_weights['regime']}
        """
        
        ax8.text(0.02, 0.98, perf_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('🔬 INTELLIGENT QUANTITATIVE TRADING SYSTEM - QLIB POWERED', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig('qlib_intelligent_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Dashboard created and saved as 'qlib_intelligent_dashboard.png'")
    
    def create_plotly_dashboard(self):
        """Create interactive Plotly dashboard for web interface"""
        print("🌐 Creating Interactive Web Dashboard...")
        
        df = self.data.dropna()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Price & Signal', 'Market Regimes', 'Strategy Performance', 
                          'Risk Metrics', 'Volatility Regime', 'Alpha Factors'),
            specs=[[{"secondary_y": True}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Price and Signal
        fig.add_trace(
            go.Scatter(x=df.index, y=df['close'], name='Price', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['final_signal'], name='Signal', 
                      line=dict(color='red'), yaxis='y2'),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Market Regimes Pie Chart
        regime_counts = df['regime_name'].value_counts()
        fig.add_trace(
            go.Pie(labels=regime_counts.index, values=regime_counts.values, name="Regimes"),
            row=1, col=2
        )
        
        # 3. Strategy Performance
        cumulative_returns = (1 + df['strategy_returns']).cumprod()
        cumulative_benchmark = (1 + df['returns']).cumprod()
        fig.add_trace(
            go.Scatter(x=df.index, y=cumulative_returns, name='Strategy', 
                      line=dict(color='green')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=cumulative_benchmark, name='Buy & Hold', 
                      line=dict(color='orange')),
            row=2, col=1
        )
        
        # 4. Risk Metrics Bar Chart
        risk_names = list(self.risk_metrics.keys())[:6]
        risk_values = [self.risk_metrics[name] for name in risk_names]
        colors = ['green' if 'sharpe' in name or 'calmar' in name else 'red' for name in risk_names]
        fig.add_trace(
            go.Bar(x=risk_names, y=risk_values, name='Risk Metrics', 
                  marker_color=colors),
            row=2, col=2
        )
        
        # 5. Volatility Regime
        fig.add_trace(
            go.Scatter(x=df.index, y=df['vol_ratio'], name='Vol Ratio', 
                      line=dict(color='purple')),
            row=3, col=1
        )
        fig.add_hline(y=1, line_dash="dash", line_color="red", row=3, col=1)
        
        # 6. Alpha Factors
        latest_factors = {}
        for name, series in self.alpha_factors.items():
            if not series.empty:
                latest_factors[name] = series.iloc[-1]
        
        factor_names = list(latest_factors.keys())[:8]
        factor_values = [latest_factors[name] for name in factor_names]
        factor_colors = ['green' if val > 0 else 'red' for val in factor_values]
        
        fig.add_trace(
            go.Bar(x=factor_values, y=factor_names, orientation='h', 
                  marker_color=factor_colors, name='Alpha Factors'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="🔬 INTELLIGENT QUANTITATIVE TRADING DASHBOARD",
            showlegend=True,
            template="plotly_dark"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_xaxes(title_text="Risk Values", row=2, col=2)
        fig.update_xaxes(title_text="Factor Values", row=3, col=2)
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Signal", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Returns", row=2, col=1)
        fig.update_yaxes(title_text="Risk Metrics", row=2, col=2)
        fig.update_yaxes(title_text="Volatility Ratio", row=3, col=1)
        fig.update_yaxes(title_text="Alpha Factors", row=3, col=2)
        
        return fig
    
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
                            for name, series in self.alpha_factors.items()}
        }
        
        return dashboard_data

# FastAPI Application
app = FastAPI(title="Qlib Trading Dashboard", version="1.0.0")

# Global engine instance
engine = None

# Create static directory
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Main dashboard page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Qlib Trading Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: white; }
            .header { text-align: center; margin-bottom: 30px; }
            .upload-section { text-align: center; margin: 20px 0; padding: 20px; background: #2a2a2a; border-radius: 10px; }
            .dashboard { display: none; }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
            .metric-card { background: #2a2a2a; padding: 15px; border-radius: 8px; text-align: center; }
            .metric-value { font-size: 24px; font-weight: bold; color: #00ff88; }
            .signal-buy { color: #00ff88; }
            .signal-sell { color: #ff4444; }
            .signal-hold { color: #ffaa00; }
            .chart-container { margin: 20px 0; }
            button { background: #00ff88; color: black; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
            button:hover { background: #00cc66; }
            input[type="file"] { margin: 10px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔬 Qlib Intelligent Trading Dashboard</h1>
            <p>Upload your CSV data to start quantitative analysis</p>
        </div>
        
        <div class="upload-section">
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="csvFile" name="file" accept=".csv" required>
                <button type="submit">Analyze Data</button>
            </form>
        </div>
        
        <div id="dashboard" class="dashboard">
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Signal</h3>
                    <div id="signal" class="metric-value">-</div>
                </div>
                <div class="metric-card">
                    <h3>Current Price</h3>
                    <div id="currentPrice" class="metric-value">-</div>
                </div>
                <div class="metric-card">
                    <h3>Regime</h3>
                    <div id="regime" class="metric-value">-</div>
                </div>
                <div class="metric-card">
                    <h3>Sharpe Ratio</h3>
                    <div id="sharpe" class="metric-value">-</div>
                </div>
                <div class="metric-card">
                    <h3>Total Return</h3>
                    <div id="totalReturn" class="metric-value">-</div>
                </div>
                <div class="metric-card">
                    <h3>Max Drawdown</h3>
                    <div id="maxDrawdown" class="metric-value">-</div>
                </div>
            </div>
            
            <div class="chart-container">
                <div id="mainChart"></div>
            </div>
            
            <div class="chart-container">
                <div id="performanceChart"></div>
            </div>
        </div>
        
        <script>
            $('#uploadForm').on('submit', function(e) {
                e.preventDefault();
                
                var formData = new FormData();
                formData.append('file', $('#csvFile')[0].files[0]);
                
                $.ajax({
                    url: '/analyze',
                    type: 'POST',
                    data: formData,
                    processData: false,
                    contentType: false,
                    success: function(response) {
                        $('#dashboard').show();
                        updateDashboard(response);
                    },
                    error: function(xhr, status, error) {
                        alert('Error: ' + error);
                    }
                });
            });
            
            function updateDashboard(data) {
                // Update metrics
                $('#signal').text(data.current_signal.direction).removeClass().addClass('metric-value signal-' + data.current_signal.direction.toLowerCase());
                $('#currentPrice').text('₹' + data.current_signal.current_price.toFixed(2));
                $('#regime').text(data.current_signal.regime);
                $('#sharpe').text(data.performance.sharpe_ratio.toFixed(2));
                $('#totalReturn').text((data.performance.total_return * 100).toFixed(2) + '%');
                $('#maxDrawdown').text((data.performance.max_drawdown * 100).toFixed(2) + '%');
                
                // Create price chart
                var priceTrace = {
                    x: data.price_data.timestamps,
                    y: data.price_data.prices,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Price',
                    line: {color: 'blue'}
                };
                
                var signalTrace = {
                    x: data.price_data.timestamps,
                    y: data.price_data.signals,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Signal',
                    line: {color: 'red'},
                    yaxis: 'y2'
                };
                
                var layout = {
                    title: 'Price and Signal',
                    xaxis: {title: 'Time'},
                    yaxis: {title: 'Price'},
                    yaxis2: {title: 'Signal', overlaying: 'y', side: 'right'},
                    template: 'plotly_dark'
                };
                
                Plotly.newPlot('mainChart', [priceTrace, signalTrace], layout);
                
                // Create performance chart
                var performanceData = [
                    {
                        x: ['Total Return', 'Sharpe Ratio', 'Hit Rate', 'Profit Factor'],
                        y: [
                            data.performance.total_return * 100,
                            data.performance.sharpe_ratio,
                            data.performance.hit_rate * 100,
                            data.performance.profit_factor
                        ],
                        type: 'bar',
                        marker: {color: ['#00ff88', '#00ff88', '#00ff88', '#00ff88']}
                    }
                ];
                
                var perfLayout = {
                    title: 'Performance Metrics',
                    xaxis: {title: 'Metrics'},
                    yaxis: {title: 'Values'},
                    template: 'plotly_dark'
                };
                
                Plotly.newPlot('performanceChart', performanceData, perfLayout);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    """Analyze uploaded CSV file"""
    global engine
    
    try:
        # Save uploaded file temporarily
        temp_file = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialize and run analysis
        engine = QuantitativeAlphaEngine(temp_file)
        engine.load_and_prepare_data()
        engine.compute_alpha_factors()
        engine.detect_market_regimes()
        engine.compute_risk_metrics()
        engine.create_composite_alpha_signal()
        engine.portfolio_optimization()
        engine.backtest_strategy()
        
        # Get dashboard data
        dashboard_data = engine.get_dashboard_data()
        
        # Clean up temp file
        os.remove(temp_file)
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/dashboard")
async def get_dashboard():
    """Get current dashboard data"""
    global engine
    if engine is None:
        raise HTTPException(status_code=400, detail="No analysis performed yet")
    
    return JSONResponse(content=engine.get_dashboard_data())

@app.get("/signal")
async def get_current_signal():
    """Get current trading signal"""
    global engine
    if engine is None:
        raise HTTPException(status_code=400, detail="No analysis performed yet")
    
    return JSONResponse(content=engine.generate_live_trading_signal())

def main():
    """Main execution function"""
    print("🚀 INTELLIGENT QUANTITATIVE TRADING SYSTEM")
    print("═" * 60)
    print("🏗️ Powered by Microsoft Qlib")
    print("🎯 Advanced Alpha Generation & Risk Management")
    print("🌐 Web Dashboard on http://localhost:8080")
    print("═" * 60)
    
    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)

if __name__ == "__main__":
    main()
