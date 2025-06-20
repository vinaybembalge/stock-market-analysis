# Tutorial: stock-market-analysis

This project focuses on analyzing stock market data.
It involves **fetching historical stock prices** using online sources,
**processing and preparing** this data (like scaling and splitting), and
then using a **neural network model** to potentially find patterns or make predictions
based on the historical performance. The model is **trained** to minimize errors in its predictions.


## Visual Overview

```mermaid
flowchart TD
    A0["Stock Data Source
"]
    A1["Historical Stock Data
"]
    A2["Neural Network Model
"]
    A3["Model Training Process
"]
    A4["Loss Function
"]
    A5["Optimizer
"]
    A6["Data Preprocessing (Scaling)
"]
    A7["Training and Testing Split
"]
    A0 -- "Fetches" --> A1
    A1 -- "Is processed by" --> A6
    A6 -- "Provides input to" --> A7
    A7 -- "Provides data to" --> A3
    A2 -- "Uses" --> A4
    A4 -- "Guides" --> A5
    A5 -- "Used by" --> A3
    A3 -- "Trains" --> A2
```

## Chapters

1. [Neural Network Model
](01_neural_network_model_.md)
2. [Stock Data Source
](02_stock_data_source_.md)
3. [Historical Stock Data
](03_historical_stock_data_.md)
4. [Data Preprocessing (Scaling)
](04_data_preprocessing__scaling__.md)
5. [Training and Testing Split
](05_training_and_testing_split_.md)
6. [Loss Function
](06_loss_function_.md)
7. [Optimizer
](07_optimizer_.md)
8. [Model Training Process
](08_model_training_process_.md)

---
