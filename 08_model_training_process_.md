# Chapter 8: Model Training Process

Welcome back! In our previous chapters, we've built a solid foundation. We understand the [Neural Network Model](01_neural_network_model_.md), where we get our [Historical Stock Data](03_historical_stock_data_.md) ([Stock Data Source](02_stock_data_source_.md)), how we prepare it using [Data Preprocessing (Scaling)](04_data_preprocessing__scaling__.md), and how we separate it into a [Training and Testing Split](05_training_and_testing_split_.md). We also learned about the [Loss Function](06_loss_function_.md), which tells the network how "wrong" it is, and the [Optimizer](07_optimizer_.md), which uses that information to figure out how to adjust the network's settings.

Now, it's time to put all these pieces together! The **Model Training Process** is where the magic happens – where the neural network actually *learns* from the data.

Think of it like teaching a complex skill, like riding a bike. You don't just tell someone the theory; they need to practice repeatedly.

*   They try (make a prediction).
*   They wobble or fall (calculate the loss/error).
*   They adjust their balance and steering based on that feedback (the optimizer updates the weights).
*   They try again (process another batch of data).

They repeat this cycle over and over until they can ride smoothly (the model's loss is minimized, and it makes good predictions).

### What is the Model Training Process?

The Model Training Process is the iterative phase where we feed the neural network the **training data** (from our [Training and Testing Split](05_training_and_testing_split_.md)) and allow it to adjust its internal parameters ([Connections (Weights)](01_neural_network_model_.md)) to minimize the [Loss Function](06_loss_function_.md) score.

It's a continuous cycle of:

1.  **Input:** Feeding historical stock data into the network.
2.  **Predict:** The network processes the data and makes a prediction.
3.  **Evaluate:** The [Loss Function](06_loss_function_.md) compares the prediction to the actual correct value and calculates the error (loss).
4.  **Optimize:** The [Optimizer](07_optimizer_.md) uses the loss to determine how to slightly adjust the network's weights to reduce the error for the *next* attempt.
5.  **Repeat:** Go back to step 1 with the next piece of data (or next batch).

This cycle is repeated many, many times.

### Key Concepts in Training

*   **Epoch:** One complete pass through the entire **training dataset**. If you have 1000 training examples and your training runs for 100 epochs, the network will have seen each example 100 times. More epochs generally mean more learning, but too many can lead to overfitting (memorizing the training data).
*   **Batch Size:** Instead of feeding one data example at a time, we usually group several examples together into a "batch". The network processes this batch, calculates the loss for the whole batch, and the [Optimizer](07_optimizer_.md) updates weights based on the average loss of the batch. This is more efficient. If you have 1000 training examples and a batch size of 10, there are 100 batches per epoch.
*   **Training Loop:** The code structure that orchestrates the repetition of processing batches for a certain number of epochs.

### How the Training Process Works

Let's trace the flow for one epoch:

1.  The training data is divided into batches (based on `batch_size`).
2.  For each batch:
    *   The batch's input data (`batch_x` in the code) is fed into the [Neural Network Model](01_neural_network_model_.md).
    *   The model produces predictions (`predictions`).
    *   These predictions and the correct answers for the batch (`batch_y`) are fed into the [Loss Function](06_loss_function_.md).
    *   The Loss Function calculates a single loss value for the batch.
    *   This loss value is sent to the [Optimizer](07_optimizer_.md).
    *   The Optimizer performs its calculation (finding the gradient) and updates the network's [Connections (Weights)](01_neural_network_model_.md).
3.  Once all batches for the epoch are processed, the epoch is complete.
4.  The process repeats for the next epoch, starting again from the first batch, but now with the updated weights.

```mermaid
sequenceDiagram
    participant Training Data Set
    participant Training Loop (Epochs)
    participant Batch of Data
    participant Neural Network Model
    participant Loss Function
    participant Optimizer
    participant Network Weights

    Training Data Set->>Training Loop (Epochs): Provide all training data
    Training Loop (Epochs)->>Training Loop (Epochs): Start Epoch 1
    Training Loop (Epochs)->>Batch of Data: Select Batch 1
    Batch of Data->>Neural Network Model: Feed input data (x)
    Batch of Data->>Loss Function: Provide correct answers (y)
    Neural Network Model->>Loss Function: Provide predictions
    Loss Function-->>Optimizer: Send calculated loss value
    Optimizer->>Network Weights: Calculate updates and adjust weights
    Training Loop (Epochs)->>Batch of Data: Select Batch 2 (Repeat for all batches in Epoch 1)
    ...
    Training Loop (Epochs)->>Training Loop (Epochs): Epoch 1 Complete
    Training Loop (Epochs)->>Training Loop (Epochs): Start Epoch 2
    ... (Repeat for all epochs)
    Training Loop (Epochs)-->>User: Training Finished!
```
This diagram shows the repeating cycle. The training loop manages the epochs and batches, and for each batch, the network, loss function, and optimizer work together to improve the weights.

### Looking at the Code (`MLPnn.py`)

The core of the training process is implemented in the main training loop within the `MLPnn.py` script. We've already loaded the training data (`training_set`, `training_labels`) and set up the network, loss, and optimizer.

Here's the section that performs the training:

```python
# ... (loading data, hyperparameters, network definition, loss, optimizer) ...

# Initialize variables (weights start with random values)
init = tf.global_variables_initializer()

print("Model successfully defined.")

# Begin training session
with tf.Session() as sess:
    # Setup for saving training logs (optional, for visualization tools like TensorBoard)
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("logs/", sess.graph)
    
    # Run the initialization operation
    sess.run(init) 
    print("Training process started.")

    # --- The Main Training Loop ---
    for epoch in range(num_epochs): # Loop for the specified number of epochs
        # Get the current batch of data for this epoch
        # The index calculation selects 'batch_size' examples
        batch_x = training_set[batch_size * epoch:batch_size * (epoch + 1)]
        batch_y = training_labels[batch_size * epoch:batch_size * (epoch + 1)]
        batch_y = np.array(batch_y).reshape(len(batch_y), 1) # Ensure batch_y is the correct shape

        # --- The Optimization Step for this Batch ---
        # We ask TensorFlow to run the 'optimizer' and calculate the 'loss'
        # Running 'optimizer' automatically triggers the forward pass, loss calculation,
        # gradient calculation, and weight update.
        _, loss_value = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})

        # Calculate average loss per sample for display
        # Adding 0.01 prevents division by zero if batch_x is somehow empty
        avg_loss = loss_value / (batch_x.shape[0] + 0.01)

        # Display progress periodically
        if epoch % log_interval == 0:
            # Print the epoch number and the current average training loss
            # (Note: The print statement calculates accuracy from loss here, which is an approximation)
            print("Epoch:", '%05d' % (epoch), "Training accuracy:", "{:.9f}".format((1 - avg_loss) * 100))
        
        # Write summary data for TensorBoard periodically
        if epoch % 50 == 0:
            summary = sess.run(merged_summary, feed_dict={x: batch_x, y: batch_y})
            summary_writer.add_summary(summary, epoch)
    
    print("Training complete!")

    # ... Test evaluation code comes after this loop ...
```

Let's break down the key parts:

1.  `sess.run(init)`: This line is crucial. Before training starts, we initialize all the variables (our network's weights) with their starting random values (as defined with `tf.random_normal` in Chapter 1).
2.  `for epoch in range(num_epochs):`: This sets up the loop that runs for the total number of epochs specified (e.g., 5000).
3.  `batch_x = training_set[batch_size * epoch:batch_size * (epoch + 1)]`: This calculates which slice of the `training_set` belongs to the current batch within the current epoch. For example, in epoch 0, batch 0 (assuming batch\_size=10, epoch=0), it takes data from index 0 to 9. In epoch 1, batch 0, it would take data from index 10 to 19 (but our code simplifies this by just taking sequential chunks across epochs, which works if `num_epochs * batch_size >= len(training_set)`). A more typical batching would iterate through the *entire* training set *within* each epoch. However, for this simple example, this sequential batching across epochs achieves the goal of repeatedly showing the network parts of the training data.
4.  `_, loss_value = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})`: This is the heart of the training step. We tell TensorFlow to execute the `optimizer` operation and fetch the current `loss` value. The `feed_dict` provides the actual data for the placeholders `x` (input) and `y` (correct output) for this specific batch. Running the `optimizer` automatically triggers the computation graph: data flows through the network, predictions are made, loss is calculated, gradients are computed, and weights are updated. The `_` means we don't care about the specific output of the `optimizer` operation itself, only the `loss_value` that is returned.
5.  `if epoch % log_interval == 0:`: This checks if the current epoch is a multiple of `log_interval` (e.g., 1000). If it is, it prints the current average loss, allowing us to monitor the training progress. We expect the loss to generally decrease over time.

This loop continues until `num_epochs` is reached. With each `sess.run([optimizer, loss], ...)` call, the network gets slightly better at predicting the values in `batch_y` from the inputs in `batch_x`.

### The Outcome of Training

After the training loop finishes, the neural network's weights are in their final, learned state. They are (hopefully!) configured in a way that allows the network to make reasonably accurate predictions on data similar to what it was trained on.

At this point, the training data has served its purpose. We then move on to evaluate the model's performance on the completely separate **test set** (`test_set`, `test_labels`) that it has never seen during training. This gives us an honest measure of how well the model has generalized and how we can expect it to perform on new, future stock data.

### Conclusion

The Model Training Process is the core learning phase for the neural network. It's a repetitive cycle driven by the [Loss Function](06_loss_function_.md) and guided by the [Optimizer](07_optimizer_.md). By repeatedly processing batches of [Training and Testing Split](05_training_and_testing_split_.md) data, calculating the error, and adjusting the network's [Connections (Weights)](01_neural_network_model_.md), the model gradually improves its ability to make accurate predictions.

Understanding how the training loop works, connecting the concepts of epochs, batches, loss, and optimization, reveals how a seemingly complex neural network learns from historical data. Once trained, the model is ready to be evaluated and potentially used for making predictions on new data.

This concludes the chapters on the core concepts behind the `stock-market-analysis` project. We've covered the building blocks of the neural network, where the data comes from, how it's prepared, and how the network learns. With this knowledge, you can dive deeper into the code and understand how these concepts are implemented to perform stock market analysis.

---
