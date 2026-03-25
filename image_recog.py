import numpy as np

from tensorflow.keras.datasets import mnist  # type: ignore
import matplotlib.pyplot as plt
np.random.seed(3)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def softmax(z):
    zexp = np.exp(z - np.max(z))
    return zexp / np.sum(zexp, axis=0, keepdims=True)
def relu_deri(z):
    return np.where(z > 0, 1, 0)

# Load MNIST data (train only, for example)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten images to 784-length vectors and normalize pixels to [0,1]

x_train = x_train.reshape(-1, 784) / 255.0

# One-hot encode labels (10 classes)
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train_onehot = one_hot(y_train)

# Network architecture
layer_dims = [784, 128, 100, 80, 32, 10]  # input layer 784, 4 hidden layers and output with 10 neurons(perceptrons)
Length = len(layer_dims)


# Initialize weights and biases
weights = {}
biases = {}
for l in range(1, Length):
    weights['w' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) *  0.1
    biases['b' + str(l)] = np.zeros((layer_dims[l], 1))

A_values = {}

Z = {}

# Training loop 
num_epoch=1000
batch_size=64
samples=x_train.shape[0]
for epoch in range (num_epoch) :
    if epoch <=700:
        learning_rate = 0.005
    else:
        learning_rate = 0.001  
    # shuffling the traning samples 
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train_onehot = y_train_onehot[indices]

    
    total_loss = 0
    correct = 0
    
    
    for start in range (0,samples,batch_size):
        
        end=start+batch_size
        x = x_train[start:end].T  # shape (784,1)
        y = y_train_onehot[start:end].T
       
        A_values["a0"] = x
    
        A=A_values["a0"]
     

      # Forward pass
        for l in range(1, Length):
            w = weights['w' + str(l)]
            b = biases['b' + str(l)]
            z = np.dot(w, A) + b
            Z["z" + str(l)] = z
            if l == Length -1 :
                A=softmax(z)
            else :
                A=relu(z)
            A_values["a" + str(l)] = A

          # Compute loss (cross-entropy)
        
        loss = -np.sum(y * np.log(A))/ x.shape[1]
        total_loss += loss*x.shape[1]

       # Back drop
        m=x_train.shape[1]
        dZ = A - y
        for l in reversed(range(1, Length)):
            A_prev = A_values["a" + str(l - 1)]
            w = weights["w" + str(l)]

            dW =(1/m)* np.dot(dZ, A_prev.T)
            db =(1/m)* np.sum(dZ, axis=1, keepdims=True)

            weights["w" + str(l)] -= learning_rate * dW
            biases["b" + str(l)] -= learning_rate * db

            if l > 1:
                dA_prev = np.dot(w.T, dZ)
                dZ = dA_prev * relu_deri(Z["z" + str(l - 1)])

        predicted = np.argmax(A,axis=0)
        actual = np.argmax(y,axis=0)
        correct += np.sum(predicted == actual)
        
    accuracy=correct/samples*100
    print(f"Epoch {epoch+1} completed. Avg Loss = {total_loss/samples:.4f} | Accuracy = {accuracy:.2f}%\n")      
    if accuracy >=99.27:
        print(f"Early stopping at epoch {epoch+1} with target accuracy reached {accuracy:.2f}%")
        break
     
        #storing the updated weights and baises
for l in range(1, Length):
    np.save(f'weights07_w{l}.npy', weights['w' + str(l)])
    np.save(f'biases02_b{l}.npy', biases['b' + str(l)])        



