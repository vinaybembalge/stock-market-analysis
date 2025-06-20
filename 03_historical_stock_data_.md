# Chapter 3: Historical Stock Data

Welcome back! In the last chapter, [Stock Data Source](02_stock_data_source_.md), we learned *where* our project gets the information it needs – from online financial sources using libraries like `yfinance`.

Now that we know how to *get* the data, let's look closely at *what* that data actually is. This is the **Historical Stock Data**, the main ingredient our neural network will use to learn and make predictions.

Think of it as a detailed diary for a stock, recorded day by day, sometimes even minute by minute. This diary holds all the essential details about what happened with the stock on each trading day.

### Why is Historical Data Important?

Our neural network, like a student learning from past examples, needs to see how a stock behaved in the past. It needs answers to questions like:

*   When the stock started the day at a certain price, where did it end up?
*   How much did the price change between the beginning and end of the day?
*   How many shares were bought and sold on a particular day?
*   Did the price hit a new high or low?

By studying thousands of entries in this "diary," the neural network can start to spot patterns or relationships between these daily events and how the stock price moved afterwards. Without this history, the network has nothing to learn from!

### What Does Historical Data Look Like?

Historical stock data is usually organized like a table, similar to a spreadsheet. Each row represents a single day (or another time interval), and each column contains a specific piece of information about the stock for that day.

This table is often stored in a format called a **Pandas DataFrame** in Python. A DataFrame is just a powerful way to handle tabular data.

Here are the most common pieces of information you'll find for each day:

| Column Name | What it means                                  |
| :---------- | :--------------------------------------------- |
| **Date**    | The specific day the trading occurred.         |
| **Open**    | The price when the market opened for the day.  |
| **High**    | The highest price the stock reached that day.  |
| **Low**     | The lowest price the stock reached that day.   |
| **Close**   | The price when the market closed for the day.  |
| **Volume**  | The total number of shares traded that day.    |
| **Adj Close** | The closing price, adjusted for things like stock splits and dividends. This is often the most important price to look at for long-term analysis as it gives a truer picture of the stock's value change. |

You saw an example of this structure in the previous chapter when we used `yfinance`.

```python
import yfinance as yf

msft = yf.Ticker("MSFT")
msft_history = msft.history(period="max")

# Display the first few rows (example output from Chapter 2)
print(msft_history.head())
```

The output shows this table format clearly:

```
                   Open        High         Low       Close     Volume  Dividends  Stock Splits
Date
1986-03-13     0.054693    0.062736    0.054693    0.060055 1031788800        0.0           0.0
1986-03-14     0.060055    0.063272    0.060055    0.062199  308160000        0.0           0.0
...
```

Each row is a different date, and you have columns for Open, High, Low, Close, Volume, and others. The `Date` column is special; it's the index that identifies each unique day's record.

### How the Project Uses This Data

In our project, this historical data (specifically the `Adj Close` price over many days) is the raw material for the neural network.

The notebook `Untitled.ipynb` shows how we might fetch this data and then select just the `Close` price (or `Adj Close`) to prepare it for the model:

```python
# Assume 'df' is the DataFrame fetched using yfinance
# (like the msft_history table we saw)

# Select only the 'Close' column
data = df.filter(['Close']) 

# Get the raw numbers from the selected column(s) as a NumPy array
dataset = data.values

# Print the first few values to see the numbers
print(dataset[:5]) 
```

This code snippet takes the full table (`df`) and narrows it down to just the 'Close' column (`data`). Then, it extracts the actual numerical values from this column into a format called a NumPy array (`dataset`), which is just a grid of numbers that's easy for calculations.

This `dataset` of historical closing prices over time becomes the sequence of numbers that the neural network will learn from.

Later in the `MLPnn.py` script, you see the project loading data directly from CSV files:

```python
import numpy as np

# Load dataset and split into training (66%) and test (33%) sets
filename = 'training_set.csv'
raw_data = open(filename, 'rt')
training_set = np.loadtxt(raw_data, delimiter=",")

filename = 'training_labels.csv'
raw_data = open(filename, 'rt')
training_labels = np.loadtxt(raw_data, delimiter=",")

# ... similar loading for test_set and test_labels
```

These CSV files (`.csv` means Comma Separated Values) are simply text files that store the historical data table. They likely contain the cleaned and prepared historical data (like the `dataset` created in the notebook after selecting and potentially processing columns) that is ready to be fed into the neural network. The `np.loadtxt` function reads these numbers from the files.

So, the flow is typically:
1.  Use a **[Stock Data Source](02_stock_data_source_.md)** (`yfinance`) to fetch the raw **Historical Stock Data** as a DataFrame.
2.  Select the specific columns (like 'Close' or 'Adj Close') and prepare the data (e.g., convert to a NumPy array, handle missing values - steps often done before saving to the CSVs).
3.  Load this prepared data from files (like the CSVs in `MLPnn.py`) into NumPy arrays or similar structures, ready for the neural network.

### Under the Hood: The Data Structure

The core idea is the table. When you request data, the data source sends back a lot of numbers organized by date and by the type of measurement (Open, High, etc.).

```mermaid
sequenceDiagram
    participant Data Source Library (e.g., yfinance)
    participant Pandas DataFrame
    participant NumPy Array (for training)

    Data Source Library (e.g., yfinance)->>Pandas DataFrame: Fetch & format raw data into table
    Note over Pandas DataFrame: Columns: Date, Open, High, Low, Close, Volume, Adj Close...

    Pandas DataFrame->>NumPy Array (for training): Select relevant columns (e.g., 'Close'), extract values
    Note over NumPy Array (for training): Just a grid of numbers, e.g., [ [val1], [val2], [val3], ... ]

    NumPy Array (for training)-->>Neural Network Input: Ready to be processed
```

The Pandas DataFrame (`msft_history`) is the organized table. The NumPy array (`dataset`, or the data loaded from CSVs) is the simpler list of numbers, usually representing a single sequence (like the daily closing prices) that the neural network needs as its input.

### Conclusion

Historical Stock Data is the essential historical record of a stock's performance, typically presented as a table (a Pandas DataFrame) with key daily metrics like Open, High, Low, Close, and Volume. It's the fundamental "experience" or "training data" that our neural network will learn from to try and understand patterns in stock movements.

We've seen how this data is fetched using libraries and how it's structured. Before we can feed this raw historical data directly into our neural network, we usually need to prepare it.

Next, we'll look at an important step called Data Preprocessing, specifically **Scaling**, to make the data suitable for the neural network.

[Next Chapter: Data Preprocessing (Scaling)](04_data_preprocessing__scaling__.md)

---
