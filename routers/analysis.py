#!/usr/bin/env python3
"""
Analysis router - Handles CSV uploads and analysis endpoints
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from datetime import datetime
import os
import sys

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.alpha_engine import QuantitativeAlphaEngine

router = APIRouter(prefix="/api", tags=["analysis"])

# Global engine instance
engine = None

@router.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    """Analyze uploaded CSV file"""
    global engine
    temp_file = None
    
    try:
        print(f"📁 Processing uploaded file: {file.filename}")
        
        # Save uploaded file temporarily
        temp_file = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"✅ File saved as: {temp_file}")
        print(f"📊 File size: {len(content)} bytes")
        
        # Initialize and run analysis
        print("🚀 Initializing QuantitativeAlphaEngine...")
        engine = QuantitativeAlphaEngine(temp_file)
        
        print("📈 Loading and preparing data...")
        engine.load_and_prepare_data()
        
        print("🧮 Computing alpha factors...")
        engine.compute_alpha_factors()
        
        print("🎯 Detecting market regimes...")
        engine.detect_market_regimes()
        
        print("⚠️ Computing risk metrics...")
        engine.compute_risk_metrics()
        
        print("📊 Creating composite alpha signal...")
        engine.create_composite_alpha_signal()
        
        print("💼 Portfolio optimization...")
        engine.portfolio_optimization()
        
        print("📈 Backtesting strategy...")
        engine.backtest_strategy()
        
        # Get dashboard data
        print("🎨 Generating dashboard data...")
        dashboard_data = engine.get_dashboard_data()
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"🗑️ Cleaned up: {temp_file}")
        
        print("✅ Analysis completed successfully!")
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"🗑️ Cleaned up temp file after error: {temp_file}")
        
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")

@router.get("/dashboard")
async def get_dashboard():
    """Get current dashboard data"""
    global engine
    if engine is None:
        raise HTTPException(status_code=400, detail="No analysis performed yet")
    
    return JSONResponse(content=engine.get_dashboard_data())

@router.get("/signal")
async def get_current_signal():
    """Get current trading signal"""
    global engine
    if engine is None:
        raise HTTPException(status_code=400, detail="No analysis performed yet")
    
    return JSONResponse(content=engine.generate_live_trading_signal())

@router.get("/status")
async def get_status():
    """Get analysis status"""
    global engine
    if engine is None:
        return JSONResponse(content={"status": "no_analysis", "message": "No analysis performed yet"})
    
    return JSONResponse(content={
        "status": "ready",
        "message": "Analysis completed successfully",
        "data_points": len(engine.data) if engine.data is not None else 0,
        "last_updated": engine.data.index[-1].strftime('%Y-%m-%d %H:%M:%S') if engine.data is not None else None
    })

@router.get("/position-dashboard")
async def get_position_dashboard():
    """Get position-focused dashboard data"""
    global engine
    if engine is None:
        raise HTTPException(status_code=400, detail="No analysis performed yet")
    
    # Import chart utilities
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.chart_utils import create_position_dashboard
    
    # Create position dashboard
    fig = create_position_dashboard(engine)
    
    # Convert to JSON for web display
    chart_json = fig.to_json()
    
    # Get position summary
    position_summary = {
        "position_type": engine.portfolio_weights['position_type'],
        "position_size": min(engine.portfolio_weights['position_size'], 10),  # Capped for testing
        "position_value": engine.portfolio_weights['position_value'],
        "current_price": engine.portfolio_weights['current_price'],
        "stop_loss": engine.portfolio_weights['stop_loss'],
        "take_profit": engine.portfolio_weights['take_profit'],
        "confidence": engine.data['signal_confidence'].iloc[-1],
        "risk_reward_ratio": engine.portfolio_weights['risk_reward_ratio'],
        "max_loss": engine.portfolio_weights['max_loss'],
        "max_profit": engine.portfolio_weights['max_profit']
    }
    
    return JSONResponse(content={
        "chart": chart_json,
        "position_summary": position_summary
    })

@router.post("/optimize")
async def optimize_parameters(request: dict):
    """Auto-optimize parameters to achieve target performance"""
    global engine
    if engine is None:
        raise HTTPException(status_code=400, detail="No analysis performed yet")
    
    try:
        # Extract parameters from request
        target_return = request.get('target_return', 0.20)
        target_sharpe = request.get('target_sharpe', 1.0)
        target_max_dd = request.get('target_max_dd', -0.10)
        param_bounds = request.get('param_bounds', {})
        
        print(f"🎯 Starting optimization with targets: Return={target_return*100:.1f}%, Sharpe={target_sharpe:.2f}, MaxDD={target_max_dd*100:.1f}%")
        print(f"📊 Parameter bounds: {param_bounds}")
        
        # Run optimization with custom bounds
        result = engine.auto_optimize_parameters(
            target_return=target_return,
            target_sharpe=target_sharpe,
            target_max_dd=target_max_dd,
            param_bounds=param_bounds
        )
        
        if result['success']:
            # Get updated dashboard data
            dashboard_data = engine.get_dashboard_data()
            
            return JSONResponse(content={
                "success": True,
                "message": "Optimization completed successfully",
                "optimal_params": result['optimal_params'],
                "performance": result['performance'],
                "dashboard_data": dashboard_data
            })
        else:
            return JSONResponse(content={
                "success": False,
                "error": result['error']
            })
            
    except Exception as e:
        print(f"❌ Optimization error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Optimization failed: {str(e)}")


@router.get("/historical-profit")
async def get_historical_profit():
    """Get historical profit analysis with Buy & Hold comparison"""
    global engine
    if engine is None:
        raise HTTPException(status_code=400, detail="No analysis performed yet")
    
    # Import chart utilities
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.chart_utils import create_historical_profit_chart
    
    # Create historical profit chart
    fig = create_historical_profit_chart(engine)
    
    # Convert to JSON for web display
    chart_json = fig.to_json()
    
    # Get comparison summary
    comparison_summary = {
        "strategy_performance": {
            "total_return": engine.backtest_results['total_return'],
            "annual_return": engine.backtest_results['annual_return'],
            "sharpe_ratio": engine.backtest_results['sharpe_ratio'],
            "max_drawdown": engine.backtest_results['max_drawdown'],
            "hit_rate": engine.backtest_results['hit_rate'],
            "profit_factor": engine.backtest_results['profit_factor']
        },
        "buy_hold_performance": engine.comparison_metrics['buy_hold'],
        "excess_performance": engine.comparison_metrics['strategy_vs_bh'],
        "qlib_features_used": len(engine.qlib_alpha_factors)
    }
    
    return JSONResponse(content={
        "chart": chart_json,
        "comparison_summary": comparison_summary
    })
