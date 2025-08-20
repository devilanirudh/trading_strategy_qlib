#!/usr/bin/env python3
"""
Main FastAPI application for Qlib Trading Dashboard
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

# Import routers
from routers import analysis, views

# Create FastAPI app
app = FastAPI(
    title="Qlib Trading Dashboard", 
    version="1.0.0",
    description="Intelligent Quantitative Trading System powered by Microsoft Qlib"
)

# Create static directory
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(views.router)
app.include_router(analysis.router)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Qlib Trading Dashboard is running"}

def main():
    """Main execution function"""
    print("🚀 INTELLIGENT QUANTITATIVE TRADING SYSTEM")
    print("═" * 60)
    print("🏗️ Powered by Microsoft Qlib")
    print("🎯 Advanced Alpha Generation & Risk Management")
    print("🌐 Web Dashboard on http://localhost:8080")
    print("📚 API Docs on http://localhost:8080/docs")
    print("═" * 60)
    
    # Run FastAPI server
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

if __name__ == "__main__":
    main()
