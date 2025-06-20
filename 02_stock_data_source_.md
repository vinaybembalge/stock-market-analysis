# Chapter 2: Stock Data Source

Welcome back! In the last chapter, [Neural Network Model](01_neural_network_model_.md), we got a first look at the "brain" of our project – the neural network – and how it's built to learn patterns from data.

But where does this crucial data come from? A neural network is useless without information to train on. For stock market analysis, this information is historical data about stock prices, trading volumes, and other market details.

Typing in years of daily stock prices manually would be impossible! Luckily, there are online services that provide this information, and special tools (Python libraries) that let our programs fetch it automatically.

Think of this **Stock Data Source** as a dedicated librarian for financial history. You tell the librarian (the library) which stock's history you want and for what dates, and they give you a neatly organized record.

In our project, the "librarians" are libraries like `yfinance` and `pandas_datareader`. They connect to online sources (like Yahoo Finance) and grab the historical stock data we need.

### Why Do We Need a Stock Data Source?

Our neural network needs to see lots of examples of how stocks have moved in the past. It needs to know:

*   What was the price of a stock at the start of the day (Open)?
*   What was its highest price during the day (High)?
*   What was its lowest price (Low)?
*   What was its price at the end of the day (Close)?
*   How many shares were traded that day (Volume)?

Getting this data for many different days and many different stocks is the first essential step. The Stock Data Source makes this possible automatically.

### Key Concepts

1.  **Stock Ticker:** This is a short, unique code used to identify a specific stock traded on an exchange. For example, Apple's ticker is `AAPL`, Microsoft's is `MSFT`, and Google's is `GOOG`. When you ask for data, you use the ticker symbol.
2.  **Historical Data:** This refers to the past records of a stock's trading activity, typically on a day-by-day basis. It includes the Open, High, Low, Close, and Volume for each day.
3.  **Python Libraries (`yfinance`, `pandas_datareader`):** These are pre-written code packages that simplify the process of getting data from online sources.
    *   `yfinance` is a popular library that can fetch data from Yahoo Finance. It's widely used because it's relatively simple and often works well.
    *   `pandas_datareader` is another library that can fetch data from various sources. `yfinance` can sometimes be used together with `pandas_datareader` (as seen in some examples) to make fetching data even more flexible.

### Using the Stock Data Source (Getting Data)

Let's look at how we use these tools to get the data. The most common way in this project, particularly shown in the provided notebook snippets, is using `yfinance`.

First, you need to tell Python you want to use the `yfinance` library:

```python
import yfinance as yf
# We often shorten 'yfinance' to 'yf' to make code shorter.
```

This line is like saying, "Okay, Python, I'm going to need the `yfinance` tool, and I'll call it `yf` from now on."

Now, let's grab some historical data for a specific stock, say Microsoft (MSFT).

```python
import yfinance as yf

# Create a Ticker object for MSFT
msft = yf.Ticker("MSFT")

# Get historical data for the maximum available period
msft_history = msft.history(period="max")

# Display the first few rows of the data
print(msft_history.head())
```

If you run this code (assuming you have `yfinance` installed), the output will look something like a table, showing the historical data:

```
                   Open        High         Low       Close     Volume  Dividends  Stock Splits
Date
1986-03-13     0.054693    0.062736    0.054693    0.060055 1031788800        0.0           0.0
1986-03-14     0.060055    0.063272    0.060055    0.062199  308160000        0.0           0.0
1986-03-17     0.062199    0.063808    0.062199    0.063272  133171200        0.0           0.0
1986-03-18     0.063272    0.063808    0.061127    0.061663   67766400        0.0           0.0
1986-03-19     0.061663    0.062199    0.060055    0.060591   47894400        0.0           0.0
```
*(Note: The exact dates and numbers might vary slightly depending on when you run it and the data source updates.)*

This table is a `pandas` DataFrame, which is a standard way to handle tabular data in Python. It's perfect for our analysis!

You can see columns for 'Open', 'High', 'Low', 'Close', 'Volume', and also 'Dividends' and 'Stock Splits' which record when those events happened. The `Date` acts as the index for this table.

You can also fetch data for a specific date range:

```python
import yfinance as yf

# Download data for SPY and AAPL between specific dates
data = yf.download("SPY AAPL", start="2017-01-01", end="2017-04-30")

# Display the data structure (showing multiple tickers)
print(data.head())
```

This might show data differently, often with the different metrics (Open, High, etc.) grouped by ticker, or with a multi-level column structure depending on the library version and parameters. The key is that you get a table of historical numbers.

```
[*********************100%%**********************]  2 of 2 completed
                      Adj Close           Close             High  ...      Low           Open        Volume
                            AAPL     SPY     AAPL      SPY  ...     AAPL     SPY     AAPL     SPY     AAPL      SPY
Date                                                          ...
2017-01-03 26.211050 208.105820 26.339286 225.330002  ... 26.111568 224.210007 25.807478 225.039993  115127600   91366500
2017-01-04 26.182030 209.315659 26.307499 226.580002  ... 26.144911 225.690002 26.262857 225.870003  84472400   78744400
2017-01-05 26.315781 209.140137 26.424999 226.529999  ... 26.248201 225.480003 26.287500 226.270004  132842000   78393900
2017-01-06 26.506620 209.775406 26.602501 227.210007  ... 26.357143 225.899994 26.437143 226.529999  127006800   71559900
2017-01-09 26.759684 209.067322 26.937500 226.460007  ... 26.642857 226.419998 26.607500 226.920003   93983600   46939700
```
*(Output may vary)*

These examples show the core idea: use a library like `yfinance` to specify the stock (or stocks) and the time period, and get the historical data back as a DataFrame (a table of numbers).

### What's Happening Under the Hood?

When you use `yfinance` or `pandas_datareader` to download data, the library isn't magic. It's essentially doing the following:

```mermaid
sequenceDiagram
    participant Your Python Code
    participant yfinance/pandas_datareader
    participant Online Financial Source (e.g., Yahoo Finance)

    Your Python Code->>yfinance/pandas_datareader: Request data for Ticker X (Dates Y-Z)
    yfinance/pandas_datareader->>Online Financial Source (e.g., Yahoo Finance): Send request (API Call)
    Online Financial Source (e.g., Yahoo Finance)-->>yfinance/pandas_datareader: Send Raw Data (CSV, JSON, etc.)
    yfinance/pandas_datareader->>yfinance/pandas_datareader: Format data into a table (Pandas DataFrame)
    yfinance/pandas_datareader-->>Your Python Code: Return the DataFrame
```

1.  Your Python code calls a function from the library (like `yf.download`).
2.  The library takes your request (ticker, dates) and translates it into a format that an online data provider (like Yahoo Finance) understands. This is usually done by sending a request to a special web address (an API endpoint).
3.  The online source receives the request, looks up the data in its databases, and sends it back to the library. The data might be in a raw format like CSV or JSON.
4.  The library receives this raw data and does the work of organizing it into a structured table format, specifically a pandas DataFrame, which is easy for Python to work with.
5.  Finally, the library returns this organized DataFrame to your Python code, ready for you to use for analysis.

The libraries handle all the complex parts: connecting to the web, sending the request correctly, handling errors if something goes wrong, and parsing the incoming data into a usable format.

### Code Snippets in the Project Notebook

The provided notebook (`Untitled.ipynb`) demonstrates using `yfinance`. You can see snippets like:

```python
import yfinance as yf

msft = yf.Ticker("MSFT")
print(msft)
```
This creates a `Ticker` object, which is like an instance representing the MSFT stock within `yfinance`.

And critically, the snippet for getting the historical price data:

```python
msft.history(period="max")
```
This is the actual call to fetch the historical data table, as we showed above. The `period="max"` argument tells it to get all available historical data for MSFT.

Another way shown is using `yf.download`:

```python
import yfinance as yf
data = yf.download("SPY AAPL", start="2017-01-01", end="2017-04-30")
```
This directly downloads data for one or more tickers (`"SPY AAPL"`) over a specified `start` and `end` date.

The notebook also shows using `pandas_datareader` after overriding it with `yfinance`:

```python
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() # <== Makes pandas_datareader use yfinance
# download dataframe using pandas_datareader
data = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2017-04-30")
```
This snippet shows how `yfinance` can integrate with `pandas_datareader` so that when you call `pdr.get_data_yahoo`, it actually uses `yfinance` behind the scenes. The result is the same: you get a pandas DataFrame with the historical data.

All these snippets point to the same core idea: asking an online source for historical stock information using easy-to-use Python functions provided by libraries like `yfinance`.

### Conclusion

In this chapter, we learned that getting the necessary historical stock data is the first crucial step in our analysis project. Instead of manually collecting this information, we use Python libraries like `yfinance` and `pandas_datareader` as our "Stock Data Source" – tools that automatically fetch the data from online financial providers and organize it neatly for us in a pandas DataFrame.

Understanding how to get this data is fundamental, as everything we do next relies on having accurate historical information.

Now that we know where the data comes from, let's take a closer look at what this historical data actually contains and how it's structured.

[Next Chapter: Historical Stock Data](03_historical_stock_data_.md)

---
