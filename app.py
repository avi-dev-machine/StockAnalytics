import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockPortfolioAnalyzer:
    def __init__(self, stocks, start_date='2020-01-01', end_date=None):
        """
        Initialize portfolio analyzer with list of stock tickers
        
        Args:
            stocks (list): List of stock ticker symbols
            start_date (str): Start date for data collection
            end_date (str): End date for data collection (defaults to current date)
        """
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = self._fetch_stock_data()
        
    def _fetch_stock_data(self):
        """Fetch historical stock data for all stocks"""
        data = {}
        for stock in self.stocks:
            try:
                stock_data = yf.download(stock, start=self.start_date, end=self.end_date)
                if not stock_data.empty:
                    data[stock] = stock_data
            except Exception as e:
                print(f"Error fetching data for {stock}: {e}")
        return data
    
    def calculate_returns(self, stock):
        """Calculate daily returns for a specific stock"""
        price_data = self.data[stock]['Adj Close']
        returns = price_data.pct_change().dropna()
        return returns
    
    def portfolio_performance(self, weights=None):
        """
        Calculate portfolio performance metrics
        
        Args:
            weights (list): Portfolio allocation weights
        
        Returns:
            dict: Performance metrics
        """
        if weights is None:
            weights = [1/len(self.stocks)] * len(self.stocks)
        
        portfolio_returns = pd.DataFrame({
            stock: self.calculate_returns(stock) for stock in self.stocks
        })
        
        weighted_returns = portfolio_returns * weights
        total_portfolio_returns = weighted_returns.sum(axis=1)
        
        return {
            'Annual Return': total_portfolio_returns.mean() * 252,
            'Annual Volatility': total_portfolio_returns.std() * np.sqrt(252),
            'Sharpe Ratio': (total_portfolio_returns.mean() * 252) / (total_portfolio_returns.std() * np.sqrt(252))
        }
    
    def predict_stock_price(self, stock, days_ahead=30):
        """
        Predict future stock prices using machine learning
        
        Args:
            stock (str): Stock ticker
            days_ahead (int): Number of days to predict
        
        Returns:
            dict: Prediction results
        """
        stock_data = self.data[stock]
        features = ['Open', 'High', 'Low', 'Volume']
        
        # Create lagged features
        for feature in features:
            stock_data[f'{feature}_Lag1'] = stock_data[feature].shift(1)
            stock_data[f'{feature}_Lag2'] = stock_data[feature].shift(2)
        
        stock_data['Target'] = stock_data['Adj Close'].shift(-1)
        stock_data.dropna(inplace=True)
        
        X = stock_data[['Open_Lag1', 'High_Lag1', 'Low_Lag1', 'Volume_Lag1', 
                        'Open_Lag2', 'High_Lag2', 'Low_Lag2', 'Volume_Lag2']]
        y = stock_data['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        
        return {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
    
    def visualize_portfolio(self):
        """Create comprehensive portfolio visualization"""
        plt.figure(figsize=(15, 10))
        plt.suptitle('Portfolio Analysis Visualization', fontsize=16)
        
        # Cumulative Returns Subplot
        plt.subplot(2, 2, 1)
        for stock in self.stocks:
            cumulative_returns = (1 + self.calculate_returns(stock)).cumprod()
            plt.plot(cumulative_returns.index, cumulative_returns.values, label=stock)
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        
        # Correlation Heatmap Subplot
        plt.subplot(2, 2, 2)
        returns_df = pd.DataFrame({
            stock: self.calculate_returns(stock) for stock in self.stocks
        })
        sns.heatmap(returns_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Stock Returns Correlation')
        
        # Risk-Return Scatter
        plt.subplot(2, 2, 3)
        returns_stats = {
            stock: {
                'Return': self.calculate_returns(stock).mean() * 252,
                'Volatility': self.calculate_returns(stock).std() * np.sqrt(252)
            } for stock in self.stocks
        }
        x = [v['Volatility'] for v in returns_stats.values()]
        y = [v['Return'] for v in returns_stats.values()]
        plt.scatter(x, y)
        plt.xlabel('Volatility')
        plt.ylabel('Annual Return')
        plt.title('Risk vs Return')
        
        plt.tight_layout()
        plt.show()

# Example Usage
def main():
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    analyzer = StockPortfolioAnalyzer(stocks)
    
    print("Portfolio Performance:")
    performance = analyzer.portfolio_performance()
    for metric, value in performance.items():
        print(f"{metric}: {value:.2%}")
    
    print("\nStock Price Predictions:")
    for stock in stocks:
        prediction = analyzer.predict_stock_price(stock)
        print(f"{stock} Prediction Metrics:")
        for metric, value in prediction.items():
            print(f"{metric}: {value:.4f}")
    
    analyzer.visualize_portfolio()

if __name__ == "__main__":
    main()