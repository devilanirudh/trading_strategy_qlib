#!/usr/bin/env python3
"""
Chart utilities for creating interactive Plotly dashboards
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_plotly_dashboard(engine):
    """Create interactive Plotly dashboard for web interface"""
    print("🌐 Creating Interactive Web Dashboard...")
    
    df = engine.data.dropna()
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=('Price & Signal', 'Market Regimes', 'Strategy Performance', 
                      'Risk Metrics', 'Volatility Regime', 'Alpha Factors',
                      'Drawdown Analysis', 'Rolling Sharpe', 'Factor IC',
                      'Position Details', 'Risk Exposure', 'Trade Analytics'),
        specs=[[{"secondary_y": True}, {"type": "pie"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "heatmap"}, {"type": "scatter"}]],
        vertical_spacing=0.06,
        horizontal_spacing=0.08
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
        row=1, col=3
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=cumulative_benchmark, name='Buy & Hold', 
                  line=dict(color='orange')),
        row=1, col=3
    )
    
    # 4. Risk Metrics Bar Chart
    risk_names = list(engine.risk_metrics.keys())[:6]
    risk_values = [engine.risk_metrics[name] for name in risk_names]
    colors = ['green' if 'sharpe' in name or 'calmar' in name else 'red' for name in risk_names]
    fig.add_trace(
        go.Bar(x=risk_names, y=risk_values, name='Risk Metrics', 
              marker_color=colors),
        row=2, col=1
    )
    
    # 5. Volatility Regime
    fig.add_trace(
        go.Scatter(x=df.index, y=df['vol_ratio'], name='Vol Ratio', 
                  line=dict(color='purple')),
        row=2, col=2
    )
    fig.add_hline(y=1, line_dash="dash", line_color="red", row=2, col=2)
    
    # 6. Alpha Factors
    latest_factors = {}
    for name, series in engine.alpha_factors.items():
        if not series.empty:
            latest_factors[name] = series.iloc[-1]
    
    factor_names = list(latest_factors.keys())[:8]
    factor_values = [latest_factors[name] for name in factor_names]
    factor_colors = ['green' if val > 0 else 'red' for val in factor_values]
    
    fig.add_trace(
        go.Bar(x=factor_values, y=factor_names, orientation='h', 
              marker_color=factor_colors, name='Alpha Factors'),
        row=2, col=3
    )
    
    # 7. Drawdown Analysis
    cumulative = (1 + df['strategy_returns']).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    fig.add_trace(
        go.Scatter(x=df.index, y=drawdown * 100, name='Drawdown %', 
                  line=dict(color='red'), fill='tonexty'),
        row=3, col=1
    )
    
    # 8. Rolling Sharpe Ratio
    rolling_sharpe = df['strategy_returns'].rolling(60).mean() / df['strategy_returns'].rolling(60).std() * np.sqrt(252 * 390)
    fig.add_trace(
        go.Scatter(x=df.index, y=rolling_sharpe, name='Rolling Sharpe', 
                  line=dict(color='orange')),
        row=3, col=2
    )
    
    # 9. Factor IC (Information Coefficient)
    ic_series = calculate_factor_ic(df, engine.alpha_factors)
    fig.add_trace(
        go.Scatter(x=df.index, y=ic_series, name='Factor IC', 
                  line=dict(color='cyan')),
        row=3, col=3
    )
    
    # 10. Position Details
    position_data = [
        engine.portfolio_weights['position_size'],
        engine.portfolio_weights['position_value'],
        engine.portfolio_weights['position_percentage']
    ]
    position_labels = ['Shares', 'Value (₹)', 'Account %']
    fig.add_trace(
        go.Bar(x=position_labels, y=position_data, name='Position', 
              marker_color=['#00ff88', '#00ff88', '#00ff88']),
        row=4, col=1
    )
    
    # 11. Risk Exposure Heatmap
    risk_exposure = create_risk_exposure_matrix(df)
    fig.add_trace(
        go.Heatmap(z=risk_exposure.values, x=risk_exposure.columns, y=risk_exposure.index,
                  colorscale='RdYlGn', name='Risk Exposure'),
        row=4, col=2
    )
    
    # 12. Trade Analytics
    trade_metrics = calculate_trade_metrics(df)
    fig.add_trace(
        go.Scatter(x=trade_metrics.index, y=trade_metrics['turnover'], name='Turnover', 
                  line=dict(color='magenta')),
        row=4, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=1600,
        title_text="🔬 INTELLIGENT QUANTITATIVE TRADING DASHBOARD",
        showlegend=True,
        template="plotly_dark"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=3)
    fig.update_xaxes(title_text="Risk Values", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_xaxes(title_text="Factor Values", row=2, col=3)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=2)
    fig.update_xaxes(title_text="Time", row=3, col=3)
    fig.update_xaxes(title_text="Position", row=4, col=1)
    fig.update_xaxes(title_text="Risk Factors", row=4, col=2)
    fig.update_xaxes(title_text="Time", row=4, col=3)
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Signal", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Returns", row=1, col=3)
    fig.update_yaxes(title_text="Risk Metrics", row=2, col=1)
    fig.update_yaxes(title_text="Volatility Ratio", row=2, col=2)
    fig.update_yaxes(title_text="Alpha Factors", row=2, col=3)
    fig.update_yaxes(title_text="Drawdown %", row=3, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=2)
    fig.update_yaxes(title_text="IC", row=3, col=3)
    fig.update_yaxes(title_text="Value", row=4, col=1)
    fig.update_yaxes(title_text="Risk Factors", row=4, col=2)
    fig.update_yaxes(title_text="Turnover", row=4, col=3)
    
    return fig

def create_simple_charts(engine):
    """Create simple charts for the web interface"""
    df = engine.data.dropna()
    
    # Price and Signal Chart
    price_chart = go.Figure()
    price_chart.add_trace(go.Scatter(
        x=df.index, 
        y=df['close'], 
        name='Price', 
        line=dict(color='blue')
    ))
    price_chart.add_trace(go.Scatter(
        x=df.index, 
        y=df['final_signal'], 
        name='Signal', 
        line=dict(color='red'),
        yaxis='y2'
    ))
    price_chart.update_layout(
        title='Price and Signal',
        xaxis_title='Time',
        yaxis_title='Price',
        yaxis2=dict(title='Signal', overlaying='y', side='right'),
        template='plotly_dark'
    )
    
    # Strategy vs Buy & Hold Comparison Chart
    comparison_chart = go.Figure()
    
    # Strategy cumulative returns
    strategy_cumulative = (1 + df['strategy_returns']).cumprod()
    comparison_chart.add_trace(go.Scatter(
        x=df.index,
        y=strategy_cumulative,
        name='Strategy',
        line=dict(color='green', width=2)
    ))
    
    # Buy & Hold cumulative returns
    bh_cumulative = (1 + df['buy_hold_returns']).cumprod()
    comparison_chart.add_trace(go.Scatter(
        x=df.index,
        y=bh_cumulative,
        name='Buy & Hold',
        line=dict(color='orange', width=2)
    ))
    
    comparison_chart.update_layout(
        title='Strategy vs Buy & Hold Performance',
        xaxis_title='Time',
        yaxis_title='Cumulative Returns',
        template='plotly_dark',
        legend=dict(x=0.02, y=0.98)
    )
    
    # Performance Comparison Chart
    performance_chart = go.Figure()
    
    # Strategy metrics
    strategy_metrics = [
        engine.backtest_results['total_return'] * 100,
        engine.backtest_results['sharpe_ratio'],
        engine.backtest_results['hit_rate'] * 100,
        engine.backtest_results['profit_factor']
    ]
    
    # Buy & Hold metrics
    bh_metrics = [
        engine.comparison_metrics['buy_hold']['total_return'] * 100,
        engine.comparison_metrics['buy_hold']['sharpe_ratio'],
        50,  # Buy & hold hit rate is typically around 50%
        1.0   # Buy & hold profit factor is typically around 1
    ]
    
    performance_chart.add_trace(go.Bar(
        x=['Total Return', 'Sharpe Ratio', 'Hit Rate', 'Profit Factor'],
        y=strategy_metrics,
        name='Strategy',
        marker_color='green'
    ))
    
    performance_chart.add_trace(go.Bar(
        x=['Total Return', 'Sharpe Ratio', 'Hit Rate', 'Profit Factor'],
        y=bh_metrics,
        name='Buy & Hold',
        marker_color='orange'
    ))
    
    performance_chart.update_layout(
        title='Performance Comparison',
        xaxis_title='Metrics',
        yaxis_title='Values',
        template='plotly_dark',
        barmode='group'
    )
    
    return price_chart, comparison_chart, performance_chart

def create_historical_profit_chart(engine):
    """Create detailed historical profit analysis chart"""
    df = engine.data.dropna()
    
    # Create subplots for comprehensive profit analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cumulative Profit Comparison', 'Rolling Returns', 
                      'Drawdown Analysis', 'Profit Distribution'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "histogram"}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Cumulative Profit Comparison
    strategy_cumulative = (1 + df['strategy_returns']).cumprod()
    bh_cumulative = (1 + df['buy_hold_returns']).cumprod()
    
    fig.add_trace(
        go.Scatter(x=df.index, y=strategy_cumulative, name='Strategy', 
                  line=dict(color='green', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=bh_cumulative, name='Buy & Hold', 
                  line=dict(color='orange', width=2)),
        row=1, col=1
    )
    
    # 2. Rolling Returns (30-period)
    strategy_rolling = df['strategy_returns'].rolling(30).mean() * 252 * 390
    bh_rolling = df['buy_hold_returns'].rolling(30).mean() * 252 * 390
    
    fig.add_trace(
        go.Scatter(x=df.index, y=strategy_rolling, name='Strategy Rolling', 
                  line=dict(color='green')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=bh_rolling, name='Buy & Hold Rolling', 
                  line=dict(color='orange')),
        row=1, col=2
    )
    
    # 3. Drawdown Analysis
    strategy_dd = (strategy_cumulative - strategy_cumulative.expanding().max()) / strategy_cumulative.expanding().max() * 100
    bh_dd = (bh_cumulative - bh_cumulative.expanding().max()) / bh_cumulative.expanding().max() * 100
    
    fig.add_trace(
        go.Scatter(x=df.index, y=strategy_dd, name='Strategy Drawdown', 
                  line=dict(color='red'), fill='tonexty'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=bh_dd, name='Buy & Hold Drawdown', 
                  line=dict(color='darkred'), fill='tonexty'),
        row=2, col=1
    )
    
    # 4. Profit Distribution
    fig.add_trace(
        go.Histogram(x=df['strategy_returns'], name='Strategy Returns', 
                    nbinsx=50, marker_color='green', opacity=0.7),
        row=2, col=2
    )
    fig.add_trace(
        go.Histogram(x=df['buy_hold_returns'], name='Buy & Hold Returns', 
                    nbinsx=50, marker_color='orange', opacity=0.7),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="📊 HISTORICAL PROFIT ANALYSIS",
        showlegend=True,
        template="plotly_dark"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Returns", row=2, col=2)
    
    fig.update_yaxes(title_text="Cumulative Returns", row=1, col=1)
    fig.update_yaxes(title_text="Annualized Returns", row=1, col=2)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    return fig

def create_position_dashboard(engine):
    """Create position-focused dashboard for desktop display"""
    df = engine.data.dropna()
    
    # Create subplots for position monitoring
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Position Size & Value', 'Stop Loss & Take Profit', 
                      'Signal Confidence', 'Risk-Reward Profile'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Position Size & Value
    position_data = [
        min(engine.portfolio_weights['position_size'], 10),  # Cap at 10 for testing
        engine.portfolio_weights['position_value'],
        engine.portfolio_weights['position_percentage']
    ]
    position_labels = ['Shares (Max 10)', 'Value (₹)', 'Account %']
    fig.add_trace(
        go.Bar(x=position_labels, y=position_data, name='Position', 
              marker_color=['#00ff88', '#00ff88', '#00ff88']),
        row=1, col=1
    )
    
    # 2. Stop Loss & Take Profit
    current_price = engine.portfolio_weights['current_price']
    stop_loss = engine.portfolio_weights['stop_loss']
    take_profit = engine.portfolio_weights['take_profit']
    
    fig.add_trace(
        go.Scatter(x=[current_price, current_price], y=[0, 1], name='Current Price',
                  line=dict(color='blue', width=3)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[stop_loss, stop_loss], y=[0, 1], name='Stop Loss',
                  line=dict(color='red', width=3)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[take_profit, take_profit], y=[0, 1], name='Take Profit',
                  line=dict(color='green', width=3)),
        row=1, col=2
    )
    
    # 3. Signal Confidence
    confidence_series = df['signal_confidence'].tail(100)
    fig.add_trace(
        go.Scatter(x=confidence_series.index, y=confidence_series, name='Confidence',
                  line=dict(color='yellow')),
        row=2, col=1
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", row=2, col=1)
    
    # 4. Risk-Reward Profile
    rr_ratio = engine.portfolio_weights['risk_reward_ratio']
    max_loss = engine.portfolio_weights['max_loss']
    max_profit = engine.portfolio_weights['max_profit']
    
    fig.add_trace(
        go.Scatter(x=['Risk', 'Reward'], y=[max_loss, max_profit], name='P&L',
                  mode='markers+lines', marker=dict(size=15, color=['red', 'green'])),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="📊 POSITION MONITORING DASHBOARD",
        showlegend=True,
        template="plotly_dark"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Position Details", row=1, col=1)
    fig.update_xaxes(title_text="Price Levels", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Risk-Reward", row=2, col=2)
    
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Level", row=1, col=2)
    fig.update_yaxes(title_text="Confidence", row=2, col=1)
    fig.update_yaxes(title_text="P&L (₹)", row=2, col=2)
    
    return fig

def calculate_factor_ic(df, alpha_factors):
    """Calculate Information Coefficient for factors"""
    ic_series = pd.Series(index=df.index, dtype=float)
    
    for i in range(60, len(df)):
        window = df.iloc[i-60:i]
        if len(window) < 30:
            continue
            
        # Calculate factor values and forward returns
        factor_values = {}
        for name, series in alpha_factors.items():
            if not series.empty and i < len(series):
                factor_values[name] = series.iloc[i]
        
        if not factor_values:
            continue
            
        # Calculate correlation with forward returns
        forward_returns = window['returns'].shift(-1).dropna()
        if len(forward_returns) > 10:
            factor_series = pd.Series(factor_values)
            ic = factor_series.corr(forward_returns.iloc[:len(factor_series)])
            ic_series.iloc[i] = ic if not pd.isna(ic) else 0
    
    return ic_series.fillna(0)

def create_risk_exposure_matrix(df):
    """Create risk exposure heatmap"""
    risk_factors = ['vol_5min', 'vol_15min', 'vol_30min', 'momentum_strength', 
                   'mean_reversion_30', 'rsi', 'bb_position', 'macd_histogram']
    
    exposure_data = []
    for factor in risk_factors:
        if factor in df.columns:
            # Calculate exposure as normalized factor value
            factor_values = df[factor].tail(20).fillna(0)
            exposure_data.append(factor_values.values)
    
    if exposure_data:
        exposure_df = pd.DataFrame(exposure_data, index=risk_factors[:len(exposure_data)])
        return exposure_df
    else:
        return pd.DataFrame()

def calculate_trade_metrics(df):
    """Calculate trading metrics"""
    metrics = pd.DataFrame(index=df.index)
    
    # Turnover (position changes)
    metrics['turnover'] = df['signal_lagged'].diff().abs().rolling(20).mean()
    
    # Volatility
    metrics['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252 * 390)
    
    return metrics
