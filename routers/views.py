#!/usr/bin/env python3
"""
Views router - Handles HTML pages and static content
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["views"])

@router.get("/", response_class=HTMLResponse)
async def root():
    """Main dashboard page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Qlib Trading Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
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
            .loading { display: none; text-align: center; margin: 20px 0; }
            .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #00ff88; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔬 Qlib Intelligent Trading Dashboard</h1>
            <p>Upload your CSV data to start quantitative analysis</p>
            <div style="margin-top: 20px;">
                <a href="/position" style="background: #00ff88; color: black; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 0 10px;">📊 Position Monitor</a>
                <a href="/historical-profit" style="background: #ff6b35; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 0 10px;">📈 Historical Profit</a>
                <a href="/docs" style="background: #333; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 0 10px;">📚 API Docs</a>
            </div>
        </div>
        
        <div class="upload-section">
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="csvFile" name="file" accept=".csv" required>
                <button type="submit">Analyze Data</button>
            </form>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing data... This may take a few moments.</p>
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
                
                // Show loading
                $('#loading').show();
                $('#dashboard').hide();
                
                $.ajax({
                    url: '/api/analyze',
                    type: 'POST',
                    data: formData,
                    processData: false,
                    contentType: false,
                    success: function(response) {
                        $('#loading').hide();
                        $('#dashboard').show();
                        updateDashboard(response);
                    },
                    error: function(xhr, status, error) {
                        $('#loading').hide();
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

@router.get("/docs", response_class=HTMLResponse)
async def docs():
    """API documentation page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Qlib Trading API Documentation</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: white; }
            .header { text-align: center; margin-bottom: 30px; }
            .endpoint { background: #2a2a2a; padding: 20px; margin: 20px 0; border-radius: 10px; }
            .method { font-weight: bold; color: #00ff88; }
            .url { font-family: monospace; background: #333; padding: 5px; border-radius: 3px; }
            .description { margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔬 Qlib Trading API Documentation</h1>
            <p>Available endpoints for the quantitative trading system</p>
        </div>
        
        <div class="endpoint">
            <div class="method">POST</div>
            <div class="url">/api/analyze</div>
            <div class="description">
                Upload and analyze a CSV file containing OHLCV data. Returns comprehensive analysis results.
            </div>
        </div>
        
        <div class="endpoint">
            <div class="method">GET</div>
            <div class="url">/api/dashboard</div>
            <div class="description">
                Get current dashboard data after analysis has been performed.
            </div>
        </div>
        
        <div class="endpoint">
            <div class="method">GET</div>
            <div class="url">/api/signal</div>
            <div class="description">
                Get the current trading signal with position details and risk management information.
            </div>
        </div>
        
        <div class="endpoint">
            <div class="method">GET</div>
            <div class="url">/api/status</div>
            <div class="description">
                Get the current status of the analysis system.
            </div>
        </div>
        
        <div class="endpoint">
            <div class="method">GET</div>
            <div class="url">/api/position-dashboard</div>
            <div class="description">
                Get position-focused dashboard with position size (1-10), stop loss, and confidence metrics.
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@router.get("/position", response_class=HTMLResponse)
async def position_monitor():
    """Position monitoring page for desktop display"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Position Monitor - Qlib Trading</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: white; }
            .header { text-align: center; margin-bottom: 30px; }
            .position-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
            .position-card { background: #2a2a2a; padding: 20px; border-radius: 10px; text-align: center; }
            .position-value { font-size: 28px; font-weight: bold; }
            .position-label { font-size: 14px; color: #ccc; margin-top: 5px; }
            .long { color: #00ff88; }
            .short { color: #ff4444; }
            .flat { color: #ffaa00; }
            .chart-container { margin: 30px 0; }
            .refresh-btn { background: #00ff88; color: black; border: none; padding: 15px 30px; border-radius: 8px; cursor: pointer; font-size: 16px; margin: 20px; }
            .refresh-btn:hover { background: #00cc66; }
            .auto-refresh { text-align: center; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 POSITION MONITORING DASHBOARD</h1>
            <p>Real-time position tracking with position size capped at 1-10 for testing</p>
        </div>
        
        <div class="auto-refresh">
            <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
            <label><input type="checkbox" id="autoRefresh" checked> Auto-refresh every 30 seconds</label>
        </div>
        
        <div id="positionSummary" class="position-grid">
            <div class="position-card">
                <div id="positionType" class="position-value">-</div>
                <div class="position-label">Position Type</div>
            </div>
            <div class="position-card">
                <div id="positionSize" class="position-value">-</div>
                <div class="position-label">Shares (Max 10)</div>
            </div>
            <div class="position-card">
                <div id="currentPrice" class="position-value">-</div>
                <div class="position-label">Current Price</div>
            </div>
            <div class="position-card">
                <div id="stopLoss" class="position-value">-</div>
                <div class="position-label">Stop Loss</div>
            </div>
            <div class="position-card">
                <div id="takeProfit" class="position-value">-</div>
                <div class="position-label">Take Profit</div>
            </div>
            <div class="position-card">
                <div id="confidence" class="position-value">-</div>
                <div class="position-label">Signal Confidence</div>
            </div>
            <div class="position-card">
                <div id="riskReward" class="position-value">-</div>
                <div class="position-label">Risk:Reward Ratio</div>
            </div>
            <div class="position-card">
                <div id="maxLoss" class="position-value">-</div>
                <div class="position-label">Max Loss (₹)</div>
            </div>
            <div class="position-card">
                <div id="maxProfit" class="position-value">-</div>
                <div class="position-label">Max Profit (₹)</div>
            </div>
        </div>
        
        <div class="chart-container">
            <div id="positionChart"></div>
        </div>
        
        <script>
            let autoRefreshInterval;
            
            function startAutoRefresh() {
                if (document.getElementById('autoRefresh').checked) {
                    autoRefreshInterval = setInterval(refreshData, 30000); // 30 seconds
                }
            }
            
            function stopAutoRefresh() {
                if (autoRefreshInterval) {
                    clearInterval(autoRefreshInterval);
                }
            }
            
            document.getElementById('autoRefresh').addEventListener('change', function() {
                if (this.checked) {
                    startAutoRefresh();
                } else {
                    stopAutoRefresh();
                }
            });
            
            function refreshData() {
                $.ajax({
                    url: '/api/position-dashboard',
                    type: 'GET',
                    success: function(response) {
                        updatePositionDisplay(response.position_summary);
                        updatePositionChart(response.chart);
                    },
                    error: function(xhr, status, error) {
                        console.error('Error fetching position data:', error);
                    }
                });
            }
            
            function updatePositionDisplay(data) {
                $('#positionType').text(data.position_type).removeClass().addClass('position-value ' + data.position_type.toLowerCase());
                $('#positionSize').text(data.position_size);
                $('#currentPrice').text('₹' + data.current_price.toFixed(2));
                $('#stopLoss').text('₹' + data.stop_loss.toFixed(2));
                $('#takeProfit').text('₹' + data.take_profit.toFixed(2));
                $('#confidence').text((data.confidence * 100).toFixed(1) + '%');
                $('#riskReward').text(data.risk_reward_ratio.toFixed(2));
                $('#maxLoss').text('₹' + data.max_loss.toFixed(0));
                $('#maxProfit').text('₹' + data.max_profit.toFixed(0));
            }
            
            function updatePositionChart(chartJson) {
                const chartData = JSON.parse(chartJson);
                Plotly.newPlot('positionChart', chartData.data, chartData.layout);
            }
            
            // Initial load
            refreshData();
            startAutoRefresh();
            
            function runOptimization() {
                const targetReturn = parseFloat(document.getElementById('targetReturn').value) / 100;
                const targetSharpe = parseFloat(document.getElementById('targetSharpe').value);
                const targetMaxDD = parseFloat(document.getElementById('targetMaxDD').value) / 100;
                
                // Collect parameter bounds
                const paramBounds = {
                    sl_multiplier: {
                        min: parseFloat(document.getElementById('slMin').value),
                        max: parseFloat(document.getElementById('slMax').value)
                    },
                    tp_multiplier: {
                        min: parseFloat(document.getElementById('tpMin').value),
                        max: parseFloat(document.getElementById('tpMax').value)
                    },
                    risk_per_trade: {
                        min: parseFloat(document.getElementById('riskMin').value) / 100,
                        max: parseFloat(document.getElementById('riskMax').value) / 100
                    },
                    min_signal_strength: {
                        min: parseFloat(document.getElementById('signalMin').value),
                        max: parseFloat(document.getElementById('signalMax').value)
                    },
                    position_size_cap: {
                        min: parseFloat(document.getElementById('posMin').value) / 100,
                        max: parseFloat(document.getElementById('posMax').value) / 100
                    }
                };
                
                const statusDiv = document.getElementById('optimizationStatus');
                statusDiv.style.display = 'block';
                statusDiv.style.background = '#444';
                statusDiv.style.color = '#fff';
                statusDiv.innerHTML = '🔄 Running optimization... This may take a few minutes.';
                
                // Disable button during optimization
                const button = event.target;
                button.disabled = true;
                button.innerHTML = '⏳ OPTIMIZING...';
                
                $.ajax({
                    url: '/api/optimize',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({
                        target_return: targetReturn,
                        target_sharpe: targetSharpe,
                        target_max_dd: targetMaxDD,
                        param_bounds: paramBounds
                    }),
                    success: function(response) {
                        if (response.success) {
                            statusDiv.style.background = '#00ff88';
                            statusDiv.style.color = '#000';
                            const newReturn = (response.performance.annual_return * 100).toFixed(2);
                            const newSharpe = response.performance.sharpe_ratio.toFixed(2);
                            const newMaxDD = (response.performance.max_drawdown * 100).toFixed(2);
                            
                            let warningMessage = '';
                            if (parseFloat(newReturn) <= -50) {
                                warningMessage = '<br><br>⚠️ <strong>WARNING:</strong> Optimization resulted in very poor performance. Consider:<br>• Widening parameter bounds<br>• Adjusting target metrics<br>• Using different market conditions';
                            }
                            
                            statusDiv.innerHTML = `
                                ✅ Optimization completed!<br>
                                📊 Optimal Parameters:<br>
                                • SL Multiplier: ${response.optimal_params.sl_multiplier.toFixed(2)}<br>
                                • TP Multiplier: ${response.optimal_params.tp_multiplier.toFixed(2)}<br>
                                • Risk per Trade: ${(response.optimal_params.risk_per_trade * 100).toFixed(1)}%<br>
                                • Min Signal: ${response.optimal_params.min_signal_strength.toFixed(2)}<br>
                                • Position Cap: ${(response.optimal_params.position_size_cap * 100).toFixed(0)}%<br>
                                <br>
                                📈 New Performance:<br>
                                • Return: <span style="color: ${parseFloat(newReturn) >= 0 ? '#00ff88' : '#ff4444'}">${newReturn}%</span><br>
                                • Sharpe: <span style="color: ${parseFloat(newSharpe) >= 0 ? '#00ff88' : '#ff4444'}">${newSharpe}</span><br>
                                • Max DD: <span style="color: #ff4444">${newMaxDD}%</span>
                                ${warningMessage}
                            `;
                            
                            // Refresh the charts with new data
                            refreshData();
                        } else {
                            statusDiv.style.background = '#ff4444';
                            statusDiv.style.color = '#fff';
                            statusDiv.innerHTML = `❌ Optimization failed: ${response.error}`;
                        }
                    },
                    error: function(xhr, status, error) {
                        statusDiv.style.background = '#ff4444';
                        statusDiv.style.color = '#fff';
                        statusDiv.innerHTML = `❌ Error: ${error}`;
                    },
                    complete: function() {
                        // Re-enable button
                        button.disabled = false;
                        button.innerHTML = '🚀 RUN AUTO-OPTIMIZATION';
                    }
                });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@router.get("/historical-profit", response_class=HTMLResponse)
async def historical_profit():
    """Historical profit analysis page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Historical Profit Analysis - Qlib Trading</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: white; }
            .header { text-align: center; margin-bottom: 30px; }
            .comparison-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
            .comparison-card { background: #2a2a2a; padding: 20px; border-radius: 10px; text-align: center; }
            .comparison-value { font-size: 24px; font-weight: bold; }
            .comparison-label { font-size: 14px; color: #ccc; margin-top: 5px; }
            .strategy { color: #00ff88; }
            .buyhold { color: #ff6b35; }
            .excess { color: #00bfff; }
            .negative { color: #ff4444; }
            .chart-container { margin: 30px 0; }
            .refresh-btn { background: #00ff88; color: black; border: none; padding: 15px 30px; border-radius: 8px; cursor: pointer; font-size: 16px; margin: 20px; }
            .refresh-btn:hover { background: #00cc66; }
            .auto-refresh { text-align: center; margin: 20px 0; }
            .qlib-info { background: #333; padding: 15px; border-radius: 8px; margin: 20px 0; text-align: center; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📈 HISTORICAL PROFIT ANALYSIS</h1>
            <p>Strategy vs Buy & Hold comparison with Qlib Alpha features</p>
        </div>
        
        <div class="optimization-panel" style="background: #333; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h3>🔧 AUTO-OPTIMIZATION PANEL</h3>
            <p>Set your target performance metrics and parameter bounds, then let AI optimize:</p>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 15px 0;">
                <div>
                    <label>Target Return (%):</label>
                    <input type="number" id="targetReturn" value="20" min="5" max="100" step="5" style="width: 100%; padding: 8px; border-radius: 5px; border: none;">
                </div>
                <div>
                    <label>Target Sharpe Ratio:</label>
                    <input type="number" id="targetSharpe" value="1.0" min="0.5" max="3.0" step="0.1" style="width: 100%; padding: 8px; border-radius: 5px; border: none;">
                </div>
                <div>
                    <label>Target Max Drawdown (%):</label>
                    <input type="number" id="targetMaxDD" value="-10" min="-50" max="-5" step="5" style="width: 100%; padding: 8px; border-radius: 5px; border: none;">
                </div>
            </div>
            
            <h4 style="margin-top: 20px; color: #00ff88;">📊 PARAMETER BOUNDS</h4>
            <p style="color: #ccc; font-size: 14px;">Set min/max values to prevent unrealistic optimization results:</p>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin: 15px 0;">
                <div style="background: #444; padding: 15px; border-radius: 8px;">
                    <label style="color: #00ff88; font-weight: bold;">Stop Loss Multiplier</label>
                    <div style="display: flex; gap: 10px; margin-top: 8px;">
                        <div style="flex: 1;">
                            <label style="font-size: 12px;">Min:</label>
                            <input type="number" id="slMin" value="0.5" min="0.1" max="2.0" step="0.1" style="width: 100%; padding: 6px; border-radius: 4px; border: none;">
                        </div>
                        <div style="flex: 1;">
                            <label style="font-size: 12px;">Max:</label>
                            <input type="number" id="slMax" value="3.0" min="1.0" max="5.0" step="0.1" style="width: 100%; padding: 6px; border-radius: 4px; border: none;">
                        </div>
                    </div>
                </div>
                
                <div style="background: #444; padding: 15px; border-radius: 8px;">
                    <label style="color: #00ff88; font-weight: bold;">Take Profit Multiplier</label>
                    <div style="display: flex; gap: 10px; margin-top: 8px;">
                        <div style="flex: 1;">
                            <label style="font-size: 12px;">Min:</label>
                            <input type="number" id="tpMin" value="1.0" min="0.5" max="3.0" step="0.1" style="width: 100%; padding: 6px; border-radius: 4px; border: none;">
                        </div>
                        <div style="flex: 1;">
                            <label style="font-size: 12px;">Max:</label>
                            <input type="number" id="tpMax" value="5.0" min="2.0" max="10.0" step="0.1" style="width: 100%; padding: 6px; border-radius: 4px; border: none;">
                        </div>
                    </div>
                </div>
                
                <div style="background: #444; padding: 15px; border-radius: 8px;">
                    <label style="color: #00ff88; font-weight: bold;">Risk per Trade (%)</label>
                    <div style="display: flex; gap: 10px; margin-top: 8px;">
                        <div style="flex: 1;">
                            <label style="font-size: 12px;">Min:</label>
                            <input type="number" id="riskMin" value="0.1" min="0.05" max="1.0" step="0.05" style="width: 100%; padding: 6px; border-radius: 4px; border: none;">
                        </div>
                        <div style="flex: 1;">
                            <label style="font-size: 12px;">Max:</label>
                            <input type="number" id="riskMax" value="2.0" min="0.5" max="5.0" step="0.1" style="width: 100%; padding: 6px; border-radius: 4px; border: none;">
                        </div>
                    </div>
                </div>
                
                <div style="background: #444; padding: 15px; border-radius: 8px;">
                    <label style="color: #00ff88; font-weight: bold;">Min Signal Strength</label>
                    <div style="display: flex; gap: 10px; margin-top: 8px;">
                        <div style="flex: 1;">
                            <label style="font-size: 12px;">Min:</label>
                            <input type="number" id="signalMin" value="0.1" min="0.05" max="0.3" step="0.05" style="width: 100%; padding: 6px; border-radius: 4px; border: none;">
                        </div>
                        <div style="flex: 1;">
                            <label style="font-size: 12px;">Max:</label>
                            <input type="number" id="signalMax" value="0.5" min="0.2" max="1.0" step="0.05" style="width: 100%; padding: 6px; border-radius: 4px; border: none;">
                        </div>
                    </div>
                </div>
                
                <div style="background: #444; padding: 15px; border-radius: 8px;">
                    <label style="color: #00ff88; font-weight: bold;">Position Size Cap (%)</label>
                    <div style="display: flex; gap: 10px; margin-top: 8px;">
                        <div style="flex: 1;">
                            <label style="font-size: 12px;">Min:</label>
                            <input type="number" id="posMin" value="10" min="5" max="30" step="5" style="width: 100%; padding: 6px; border-radius: 4px; border: none;">
                        </div>
                        <div style="flex: 1;">
                            <label style="font-size: 12px;">Max:</label>
                            <input type="number" id="posMax" value="100" min="20" max="200" step="10" style="width: 100%; padding: 6px; border-radius: 4px; border: none;">
                        </div>
                    </div>
                </div>
            </div>
            
            <button onclick="runOptimization()" style="background: #00ff88; color: black; border: none; padding: 12px 25px; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: bold; margin-top: 15px;">
                🚀 RUN AUTO-OPTIMIZATION
            </button>
            
            <div id="optimizationStatus" style="margin-top: 15px; padding: 10px; border-radius: 5px; display: none;"></div>
        </div>
        
        <div class="qlib-info">
            <h3>🔬 Qlib Alpha Features Used</h3>
            <p>Alpha158: Price momentum, volume trends, volatility metrics</p>
            <p>Alpha360: Enhanced price/volume features, microstructure indicators</p>
            <p>Custom Features: Advanced momentum, mean reversion, ML-inspired features</p>
        </div>
        
        <div class="auto-refresh">
            <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Analysis</button>
            <label><input type="checkbox" id="autoRefresh" checked> Auto-refresh every 60 seconds</label>
        </div>
        
        <div id="comparisonSummary" class="comparison-grid">
            <div class="comparison-card">
                <div id="strategyReturn" class="comparison-value strategy">-</div>
                <div class="comparison-label">Strategy Total Return</div>
            </div>
            <div class="comparison-card">
                <div id="buyHoldReturn" class="comparison-value buyhold">-</div>
                <div class="comparison-label">Buy & Hold Return</div>
            </div>
            <div class="comparison-card">
                <div id="excessReturn" class="comparison-value excess">-</div>
                <div class="comparison-label">Excess Return</div>
            </div>
            <div class="comparison-card">
                <div id="strategySharpe" class="comparison-value strategy">-</div>
                <div class="comparison-label">Strategy Sharpe</div>
            </div>
            <div class="comparison-card">
                <div id="buyHoldSharpe" class="comparison-value buyhold">-</div>
                <div class="comparison-label">Buy & Hold Sharpe</div>
            </div>
            <div class="comparison-card">
                <div id="informationRatio" class="comparison-value excess">-</div>
                <div class="comparison-label">Information Ratio</div>
            </div>
            <div class="comparison-card">
                <div id="strategyDrawdown" class="comparison-value strategy">-</div>
                <div class="comparison-label">Strategy Max DD</div>
            </div>
            <div class="comparison-card">
                <div id="buyHoldDrawdown" class="comparison-value buyhold">-</div>
                <div class="comparison-label">Buy & Hold Max DD</div>
            </div>
            <div class="comparison-card">
                <div id="qlibFeatures" class="comparison-value excess">-</div>
                <div class="comparison-label">Qlib Features Used</div>
            </div>
        </div>
        
        <div class="chart-container">
            <div id="profitChart"></div>
        </div>
        
        <script>
            let autoRefreshInterval;
            
            function startAutoRefresh() {
                if (document.getElementById('autoRefresh').checked) {
                    autoRefreshInterval = setInterval(refreshData, 60000); // 60 seconds
                }
            }
            
            function stopAutoRefresh() {
                if (autoRefreshInterval) {
                    clearInterval(autoRefreshInterval);
                }
            }
            
            document.getElementById('autoRefresh').addEventListener('change', function() {
                if (this.checked) {
                    startAutoRefresh();
                } else {
                    stopAutoRefresh();
                }
            });
            
            function refreshData() {
                $.ajax({
                    url: '/api/historical-profit',
                    type: 'GET',
                    success: function(response) {
                        updateComparisonDisplay(response.comparison_summary);
                        updateProfitChart(response.chart);
                    },
                    error: function(xhr, status, error) {
                        console.error('Error fetching historical profit data:', error);
                    }
                });
            }
            
            function updateComparisonDisplay(data) {
                // Strategy Return with color coding
                const strategyReturn = (data.strategy_performance.total_return * 100).toFixed(2);
                $('#strategyReturn').text(strategyReturn + '%');
                $('#strategyReturn').removeClass('strategy negative').addClass(
                    parseFloat(strategyReturn) >= 0 ? 'strategy' : 'negative'
                );
                
                // Buy & Hold Return with color coding
                const buyHoldReturn = (data.buy_hold_performance.total_return * 100).toFixed(2);
                $('#buyHoldReturn').text(buyHoldReturn + '%');
                $('#buyHoldReturn').removeClass('buyhold negative').addClass(
                    parseFloat(buyHoldReturn) >= 0 ? 'buyhold' : 'negative'
                );
                
                // Excess Return with color coding
                const excessReturn = (data.excess_performance.excess_return * 100).toFixed(2);
                $('#excessReturn').text(excessReturn + '%');
                $('#excessReturn').removeClass('excess negative').addClass(
                    parseFloat(excessReturn) >= 0 ? 'excess' : 'negative'
                );
                
                // Sharpe ratios with color coding
                const strategySharpe = data.strategy_performance.sharpe_ratio.toFixed(2);
                $('#strategySharpe').text(strategySharpe);
                $('#strategySharpe').removeClass('strategy negative').addClass(
                    parseFloat(strategySharpe) >= 0 ? 'strategy' : 'negative'
                );
                
                const buyHoldSharpe = data.buy_hold_performance.sharpe_ratio.toFixed(2);
                $('#buyHoldSharpe').text(buyHoldSharpe);
                $('#buyHoldSharpe').removeClass('buyhold negative').addClass(
                    parseFloat(buyHoldSharpe) >= 0 ? 'buyhold' : 'negative'
                );
                
                const infoRatio = data.excess_performance.information_ratio.toFixed(2);
                $('#informationRatio').text(infoRatio);
                $('#informationRatio').removeClass('excess negative').addClass(
                    parseFloat(infoRatio) >= 0 ? 'excess' : 'negative'
                );
                
                // Drawdowns (always negative, so keep original colors)
                $('#strategyDrawdown').text((data.strategy_performance.max_drawdown * 100).toFixed(2) + '%');
                $('#buyHoldDrawdown').text((data.buy_hold_performance.max_drawdown * 100).toFixed(2) + '%');
                $('#qlibFeatures').text(data.qlib_features_used);
            }
            
            function updateProfitChart(chartJson) {
                const chartData = JSON.parse(chartJson);
                Plotly.newPlot('profitChart', chartData.data, chartData.layout);
            }
            
            // Initial load
            refreshData();
            startAutoRefresh();
            
            function runOptimization() {
                const targetReturn = parseFloat(document.getElementById('targetReturn').value) / 100;
                const targetSharpe = parseFloat(document.getElementById('targetSharpe').value);
                const targetMaxDD = parseFloat(document.getElementById('targetMaxDD').value) / 100;
                
                // Collect parameter bounds
                const paramBounds = {
                    sl_multiplier: {
                        min: parseFloat(document.getElementById('slMin').value),
                        max: parseFloat(document.getElementById('slMax').value)
                    },
                    tp_multiplier: {
                        min: parseFloat(document.getElementById('tpMin').value),
                        max: parseFloat(document.getElementById('tpMax').value)
                    },
                    risk_per_trade: {
                        min: parseFloat(document.getElementById('riskMin').value) / 100,
                        max: parseFloat(document.getElementById('riskMax').value) / 100
                    },
                    min_signal_strength: {
                        min: parseFloat(document.getElementById('signalMin').value),
                        max: parseFloat(document.getElementById('signalMax').value)
                    },
                    position_size_cap: {
                        min: parseFloat(document.getElementById('posMin').value) / 100,
                        max: parseFloat(document.getElementById('posMax').value) / 100
                    }
                };
                
                const statusDiv = document.getElementById('optimizationStatus');
                statusDiv.style.display = 'block';
                statusDiv.style.background = '#444';
                statusDiv.style.color = '#fff';
                statusDiv.innerHTML = '🔄 Running optimization... This may take a few minutes.';
                
                // Disable button during optimization
                const button = event.target;
                button.disabled = true;
                button.innerHTML = '⏳ OPTIMIZING...';
                
                $.ajax({
                    url: '/api/optimize',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({
                        target_return: targetReturn,
                        target_sharpe: targetSharpe,
                        target_max_dd: targetMaxDD,
                        param_bounds: paramBounds
                    }),
                    success: function(response) {
                        if (response.success) {
                            statusDiv.style.background = '#00ff88';
                            statusDiv.style.color = '#000';
                            statusDiv.innerHTML = `
                                ✅ Optimization completed!<br>
                                📊 Optimal Parameters:<br>
                                • SL Multiplier: ${response.optimal_params.sl_multiplier.toFixed(2)}<br>
                                • TP Multiplier: ${response.optimal_params.tp_multiplier.toFixed(2)}<br>
                                • Risk per Trade: ${(response.optimal_params.risk_per_trade * 100).toFixed(1)}%<br>
                                • Min Signal: ${response.optimal_params.min_signal_strength.toFixed(2)}<br>
                                • Position Cap: ${(response.optimal_params.position_size_cap * 100).toFixed(0)}%<br>
                                <br>
                                📈 New Performance:<br>
                                • Return: ${(response.performance.annual_return * 100).toFixed(2)}%<br>
                                • Sharpe: ${response.performance.sharpe_ratio.toFixed(2)}<br>
                                • Max DD: ${(response.performance.max_drawdown * 100).toFixed(2)}%
                            `;
                            
                            // Refresh the charts with new data
                            refreshData();
                        } else {
                            statusDiv.style.background = '#ff4444';
                            statusDiv.style.color = '#fff';
                            statusDiv.innerHTML = `❌ Optimization failed: ${response.error}`;
                        }
                    },
                    error: function(xhr, status, error) {
                        statusDiv.style.background = '#ff4444';
                        statusDiv.style.color = '#fff';
                        statusDiv.innerHTML = `❌ Error: ${error}`;
                    },
                    complete: function() {
                        // Re-enable button
                        button.disabled = false;
                        button.innerHTML = '🚀 RUN AUTO-OPTIMIZATION';
                    }
                });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
