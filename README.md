# OrderBook Analytics Engine

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Production Ready](https://img.shields.io/badge/status-production--ready-success)]()

A high-performance, production-ready analytics engine for real-time monitoring of order book health, market microstructure, and execution quality in crypto and derivatives markets.

## ✨ Features

### 🏗️ Core Architecture
- **Layered Design**: Clean separation between data, analytics, and visualization
- **O(log n) Operations**: Efficient order book management
- **Modular Components**: Easy to extend and customize
- **Production Ready**: Battle-tested with comprehensive error handling

### 📊 Analytics Capabilities
- **Spread Analysis**: Absolute/relative spreads, price impact, statistical analysis
- **Depth Analysis**: Volume imbalance, liquidity scores, order book slope, resilience
- **Liquidity Analysis**: Kyle's lambda, Amihud ratio, market impact models
- **Execution Quality**: VWAP estimation, slippage calculation, market impact

### 📈 Visualization
- **Interactive Dashboards**: Plotly-based real-time visualizations
- **Multi-panel Analytics**: Comprehensive market overview
- **Export Ready**: HTML, PNG, PDF outputs
- **Customizable Themes**: Light/dark mode support

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/orderbook-analytics.git
cd orderbook-analytics

# Install dependencies
pip install -r requirements.txt

# Run demo
python main.py