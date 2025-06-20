# Chapter 4: Data Preprocessing (Scaling)

Welcome back! In the last chapter, we explored [Historical Stock Data](03_historical_stock_data_.md) and saw how we get raw historical information, like daily closing prices, from a [Stock Data Source](02_stock_data_source_.md). We end up with a long list of numbers representing stock prices over time.

Now, imagine you're trying to compare apples and oranges, or maybe miles and kilometers. You wouldn't just add them together directly, right? You'd first need to convert one to the other so they're in the same units.

Similarly, before we feed our historical stock data into the [Neural Network Model](01_neural_network_model_.md) we discussed in Chapter 1, we need to prepare it. Stock prices can range from a few dollars to hundreds or even thousands of dollars. Other data we might include (like trading volume, which can be millions or billions) would have totally different ranges.

Neural networks, especially the type we're using, often perform better when all the input data is within a similar, small range. This is where **Data Preprocessing**, specifically **Scaling**, comes in.

### What is Scaling?

Scaling is a technique used to transform the range of your data. It adjusts the numbers in your dataset so they all fall within a specified range, commonly between 0 and 1, or sometimes between -1 and 1.

Why do we do this?

1.  **Fairness for the Network:** Neural networks learn by adjusting "weights" ([Connections (Weights)](01_neural_network_model_.md)). If one feature (like stock price, which might be $200) has a much larger range than another feature (like a calculated indicator that's always between 0 and 1), the large numbers can dominate the learning process. Scaling gives all features a fair chance to influence the model.
2.  **Faster Learning:** Many neural network **[Optimizer](07_optimizer_.md)** algorithms (which we'll cover later) work more efficiently and converge faster when the data is scaled.
3.  **Activation Functions:** Some common activation functions used in neural networks (like the sigmoid function we saw in Chapter 1, `tf.nn.sigmoid`) output values between 0 and 1. Having input data scaled to a similar range can be helpful.

In short, scaling doesn't change the *shape* or *relationships* within your data (e.g., if one price was higher than another before scaling, it will still be relatively higher after scaling), but it changes the *values* to fit into a smaller, standardized box.

### Scaling Our Stock Data

Our raw historical data might look like a series of closing prices: [50.25, 51.10, 50.80, ..., 350.50, 352.00, ...]. We want to transform these numbers so they are all between 0 and 1.

A very common way to do this is using **Min-Max Scaling**. The formula is quite simple:

`Scaled Value = (Original Value - Minimum Value) / (Maximum Value - Minimum Value)`

Let's say your stock prices range from a minimum of $50 to a maximum of $400 over the period you're looking at.

*   A price of $50 would become: (50 - 50) / (400 - 50) = 0 / 350 = 0
*   A price of $400 would become: (400 - 50) / (400 - 50) = 350 / 350 = 1
*   A price of $200 would become: (200 - 50) / (400 - 50) = 150 / 350 = 0.428...

All the prices between $50 and $400 will be transformed into values between 0 and 1.

### Using `sklearn` for Scaling

We use the `scikit-learn` library (often called `sklearn`), a popular tool for machine learning in Python, to perform scaling easily. Specifically, we'll use the `MinMaxScaler`.

First, you need to import the tool:

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np # We'll need this for example data
```

Now, let's see it in action with a tiny example:

```python
# Imagine this is a small piece of our stock price data
# It needs to be in a specific format (like a list of lists, or 2D array)
sample_data = np.array([[50.25], [65.50], [80.10], [95.99]])

# Create a scaler object
# We tell it we want the resulting values to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))

# Apply the scaling
# .fit_transform() calculates the min/max of the data (fit)
# and then applies the scaling formula (transform)
scaled_sample = scaler.fit_transform(sample_data)

# Print the original and scaled data
print("Original data:")
print(sample_data)
print("\nScaled data (0 to 1):")
print(scaled_sample)
```

Output:

```
Original data:
[[50.25]
 [65.5 ]
 [80.1 ]
 [95.99]]

Scaled data (0 to 1):
[[0.        ]
 [0.33639761]
 [0.66864393]
 [1.        ]]
```

Notice how the smallest value (50.25) becomes 0 and the largest (95.99) becomes 1. The values in between are scaled proportionally.

### Under the Hood: `fit` and `transform`

The `MinMaxScaler` (and other scalers in `sklearn`) typically has two main steps:

1.  **`fit()`:** This step *learns* the parameters needed for scaling from your data. For `MinMaxScaler`, it calculates the minimum and maximum value in the data you give it.
2.  **`transform()`:** This step *applies* the scaling formula using the parameters learned during the `fit` step.

Combining them with `fit_transform()` is common when you first process your data.

```mermaid
sequenceDiagram
    participant Raw Stock Data (e.g., Prices)
    participant MinMaxScaler (fit)
    participant MinMaxScaler (transform)
    participant Scaled Data (0 to 1)

    Raw Stock Data (e.g., Prices)->>MinMaxScaler (fit): Feed data to learn min/max
    MinMaxScaler (fit)-->>MinMaxScaler (fit): Calculate min value, max value
    MinMaxScaler (fit)-->>MinMaxScaler (transform): Store min/max values

    Raw Stock Data (e.g., Prices)->>MinMaxScaler (transform): Feed same data again
    MinMaxScaler (transform)->>MinMaxScaler (transform): Apply formula: (Value - Min) / (Max - Min)
    MinMaxScaler (transform)-->>Scaled Data (0 to 1): Output the transformed numbers
```

Why is `fit` separate from `transform`? This becomes crucial when you have a [Training and Testing Split](05_training_and_testing_split_.md) (which we'll cover next). You **only** `fit` the scaler on your *training* data to learn its min/max. Then, you use the *same* scaler object to `transform` both the training data *and* the test data. This prevents "data leakage" where information about the test set's range influences the training process.

### Connecting to the Project Code

The `Untitled.ipynb` notebook demonstrates this scaling step. After fetching the historical data and selecting the 'Close' price column ([Historical Stock Data](03_historical_stock_data_.md)), the code prepares it:

```python
# Assume 'dataset' is the numpy array of closing prices
# from the previous chapter (e.g., [[50.25], [51.10], ...])

# Import the scaler
from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler object targeting the range [0, 1]
scaler = MinMaxScaler(feature_range=(0,1))

# Fit the scaler to the data and transform the data
scaled_data = scaler.fit_transform(dataset)

# Print the first few scaled values
print(scaled_data[:5])
```

This snippet directly applies the `MinMaxScaler` to the `dataset` (which is a NumPy array of historical closing prices). The `scaled_data` variable now holds the same price data, but with all values adjusted to be between 0 and 1. This `scaled_data` is what will eventually be used to train the neural network.

The notebook also shows using the *same* `scaler` object later to reverse the process for the final predictions:

```python
# ... (model prediction code) ...
predictions = model.predict(x_test) 

# Now, convert the predictions back to the original price range
predictions = scaler.inverse_transform(predictions) 
```

This `inverse_transform` step is vital. The neural network makes predictions on the *scaled* data, so its output will be numbers between 0 and 1. To understand what those numbers mean in terms of actual stock prices, we use the *same* `scaler` object to convert them back to their original scale.

### Conclusion

Data Preprocessing, specifically Scaling, is a vital step before feeding raw historical data into a neural network. It transforms the numerical values into a consistent range (like 0 to 1) using tools like `sklearn`'s `MinMaxScaler`. This prevents larger values from dominating the learning process and helps the network train more effectively and quickly.

We've seen how to apply scaling using `fit_transform` and how the project's code implements this step. We also touched upon the importance of `inverse_transform` to get predictions back into a usable format (actual prices).

With our data now scaled and ready, the next crucial step is to divide it into groups: one for the neural network to learn from (training data) and one to check how well it learned (testing data).

[Next Chapter: Training and Testing Split](05_training_and_testing_split_.md)

---
