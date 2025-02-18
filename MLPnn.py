import tensorflow as tf
import numpy as np

# Load dataset and split into training (66%) and test (33%) sets
filename = 'training_set.csv'
raw_data = open(filename, 'rt')
training_set = np.loadtxt(raw_data, delimiter=",")

filename = 'training_labels.csv'
raw_data = open(filename, 'rt')
training_labels = np.loadtxt(raw_data, delimiter=",")

filename = 'test_set.csv'
raw_data = open(filename, 'rt')
test_set = np.loadtxt(raw_data, delimiter=",")

filename = 'test_labels.csv'
raw_data = open(filename, 'rt')
test_labels = np.loadtxt(raw_data, delimiter=",")

print("Dataset successfully loaded.")

# Hyperparameters
learning_rate = 0.01 
batch_size = 10
num_epochs = 5000
log_interval = 1000

# Neural Network Specifications
num_neurons_1 = 64    # First hidden layer
num_neurons_2 = 32    # Second hidden layer
num_neurons_3 = 16    # Third hidden layer
num_features = 50     # Input layer size

# Define placeholders for input and output
with tf.name_scope("inputs"):
    x = tf.placeholder(tf.float64, shape=[None, num_features], name="input_data")
    y = tf.placeholder(tf.float64, shape=[None, 1], name="output_labels")

print("Initializing the neural network model.")

# Define the neural network architecture
def mlp_model(x, weights):
    with tf.name_scope("layers"):
        with tf.name_scope("weights"):
            weights['h1'] = tf.Variable(tf.random_normal([num_features, num_neurons_1], dtype=np.float64), name="W1")
            tf.summary.histogram("layer1_weights", weights['h1'])
            weights['h2'] = tf.Variable(tf.random_normal([num_neurons_1, num_neurons_2], dtype=np.float64), name="W2")
            tf.summary.histogram("layer2_weights", weights['h2'])
            weights['h3'] = tf.Variable(tf.random_normal([num_neurons_2, num_neurons_3], dtype=np.float64), name="W3")
            tf.summary.histogram("layer3_weights", weights['h3'])
            weights['out'] = tf.Variable(tf.random_normal([num_neurons_3, 1], dtype=np.float64), name="W4")
            tf.summary.histogram("output_weights", weights['out'])
        
        # Compute activations for each layer
        with tf.name_scope("activations"):
            layer1 = tf.nn.sigmoid(tf.matmul(x, weights['h1']))
            tf.summary.histogram("layer1_output", layer1)
            layer2 = tf.nn.sigmoid(tf.matmul(layer1, weights['h2']))
            tf.summary.histogram("layer2_output", layer2)
            layer3 = tf.nn.sigmoid(tf.matmul(layer2, weights['h3']))
            tf.summary.histogram("layer3_output", layer3)
            output_layer = tf.nn.sigmoid(tf.matmul(layer3, weights['out']))
            tf.summary.histogram("final_output", output_layer)
        
    return output_layer

# Initialize weight variables
weights = {
    'h1': tf.Variable(tf.random_normal([num_features, num_neurons_1], dtype=np.float64)),       
    'h2': tf.Variable(tf.random_normal([num_neurons_1, num_neurons_2], dtype=np.float64)),
    'h3': tf.Variable(tf.random_normal([num_neurons_2, num_neurons_3], dtype=np.float64)),
    'out': tf.Variable(tf.random_normal([num_neurons_3, 1], dtype=np.float64))
}

# Construct the model
predictions = mlp_model(x, weights)

# Define loss function and optimization method
with tf.name_scope("loss_function"):
    loss = tf.nn.l2_loss(predictions - y, name="mean_squared_error")
    tf.summary.scalar("training_loss", loss)

with tf.name_scope("optimizer"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()

print("Model successfully defined.")

# Begin training session
with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("logs/", sess.graph)
    
    sess.run(init)
    print("Training process started.")

    # Training loop
    for epoch in range(num_epochs):
        batch_x = training_set[batch_size * epoch:batch_size * (epoch + 1)]
        batch_y = training_labels[batch_size * epoch:batch_size * (epoch + 1)]
        batch_y = np.array(batch_y).reshape(len(batch_y), 1)
        _, loss_value = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})

        avg_loss = loss_value / (batch_x.shape[0] + 0.01)

        # Display progress at intervals
        if epoch % log_interval == 0:
            print("Epoch:", '%05d' % (epoch), "Training accuracy:", "{:.9f}".format((1 - avg_loss) * 100))
        
        if epoch % 50 == 0:
            summary = sess.run(merged_summary, feed_dict={x: batch_x, y: batch_y})
            summary_writer.add_summary(summary, epoch)
    
    print("Training complete!")

    # Evaluate test performance
    test_loss = tf.nn.l2_loss(predictions - y, name="test_loss") / test_set.shape[0]
    test_labels = np.array(test_labels).reshape(len(test_labels), 1)
    print("Test Accuracy:", 100 - test_loss.eval({x: test_set, y: test_labels}) * 100, "%")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
