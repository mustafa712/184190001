import sys
import os
import numpy as np
import pandas as pd

# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)

NUM_FEATS = 90

class Net(object):
    '''
    '''

    def __init__(self, num_layers, num_units):
        '''
        Initialize the neural network.
        Create weights and biases.

        Here, we have provided an example structure for the weights and biases.
        It is a list of weight and bias matrices, in which, the
        dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
        weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
        biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]

        Please note that this is just an example.
        You are free to modify or entirely ignore this initialization as per your need.
        Also you can add more state-tracking variables that might be useful to compute
        the gradients efficiently.


        Parameters
        ----------
            num_layers : Number of HIDDEN layers.
            num_units : Number of units in each Hidden layer.
        '''
        self.num_layers = num_layers
        self.num_units = num_units

        self.biases = []
        self.weights = []
        self.activation = []
        self.z = []
        for i in range(num_layers):

            if i==0:
                # Input layer
                self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
            else:
                # Hidden layer
                self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))

            self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

        # Output layer
        self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
        self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))
        self.output = None

    def __call__(self, X):
        '''
        Forward propagate the input X through the network,
        and return the output.

        Note that for a classification task, the output layer should
        be a softmax layer. So perform the computations accordingly

        Parameters
        ----------
            X : Input to the network, numpy array of shape m x d
        Returns
        ----------
            y : Output of the network, numpy array of shape m x 1
        '''
        self.activation = []
        self.z = []
        relu = np.vectorize(lambda _ : max(0, _))
        for i in range(self.num_layers):
            if i == 0:
                values = np.matmul(X, self.weights[i])\
                                   + self.biases[i].transpose()
            else:
                values = np.matmul(self.activation[i - 1], self.weights[i])\
                                   + self.biases[i].transpose()
            #print("Values", values)
            self.z.append(values)
            self.activation.append(relu(values))

        self.output = np.matmul(self.activation[-1], self.weights[-1])\
                                       + self.biases[-1].transpose()
        return self.output

    def backward(self, X, y, lamda):
        '''
        Compute and return gradients loss with respect to weights and biases.
        (dL/dW and dL/db)

        Parameters
        ----------
            X : Input to the network, numpy array of shape m x d
            y : Output of the network, numpy array of shape m x 1
            lamda : Regularization parameter.

        Returns
        ----------
            del_W : derivative of loss w.r.t. all weight values (a list of matrices).
            del_b : derivative of loss w.r.t. all bias values (a list of vectors).

        Hint: You need to do a forward pass before performing backward pass.
        '''
        twoyhat_y = 2*(self.output - y)
        m = len(y)

        dw = np.sum(twoyhat_y*self.activation[-1], axis=0)/m
        del_W = [dw.reshape(self.num_units, 1)]
        db = np.sum(twoyhat_y)/m
        del_b = [db.reshape(1, 1)]
        relu_d = np.vectorize(lambda x : 1 if x > 0 else 0)

        for i in range(self.num_layers - 1, -1, -1):
            delta = np.matmul(self.weights[i+1], del_b[0]).transpose() * relu_d(self.z[i])
            del_b = [(np.sum(delta, axis=0)/m).reshape(self.biases[i].shape)] + del_b
            if i == 0:
                del_W = [np.matmul(X.transpose(), delta)/m] + del_W
            else:
                del_W = [np.matmul(self.activation[i - 1].transpose(), delta)/m] + del_W

        for i in range(self.num_layers):
            del_W[i] += lamda*self.weights[i]
            del_b[i] += lamda*self.biases[i]

        del_W[-1] += lamda*self.weights[-1]
        del_b[-1] += lamda*self.biases[-1]

        #print("db", del_b, "dw", del_W)

        return del_W, del_b

    def save(self, fname):
        with open(fname, "w") as f:
            f.write(f"{self.num_layers}\n")
            f.write(f"{self.num_units}\n")
            for i in range(self.num_layers):
                if i == 0:
                    for r in range(NUM_FEATS):
                        for c in range(self.num_units):
                            f.write(f"{self.weights[i][r, c]},")
                        f.write("\n")
                else:
                    for r in range(self.num_units):
                        for c in range(self.num_units):
                            f.write(f"{self.weights[i][r, c]},")
                        f.write("\n")
                for j in range(self.num_units):
                    f.write(f"{self.biases[i][j, 0]},")
                f.write("\n")
            for j in range(self.num_units):
                f.write(f"{self.weights[-1][j, 0]},")
            f.write("\n")
            f.write(f"{self.biases[-1][0, 0]}")

    def set(self, fname):
        with open(fname, "r") as f:
            line = f.readline()
            self.num_layers = int(line[:-1])
            line = f.readline()
            self.num_units = int(line[:-1])
            self.weights = []
            self.biases = []
            for i in range(self.num_layers):
                if i == 0:
                    for r in range(NUM_FEATS):
                        line = f.readline().split(",")
                        w = np.zeros(shape=(NUM_FEATS, self.num_units))
                        for c in range(self.num_units):
                            w[r, c] = float(line[c])
                    self.weights.append(w)
                else:
                    for r in range(self.num_units):
                        line = f.readline().split(",")
                        w = np.zeros(shape=(self.num_units, self.num_units))
                        for c in range(self.num_units):
                            w[r, c] = float(line[c])
                    self.weights.append(w)
                line = f.readline().split(",")
                b = np.zeros(shape=(self.num_units, 1))
                for j in range(self.num_units):
                    b[j, 0] = float(line[j])
                self.biases.append(b)
            line = f.readline.split(",")
            w = np.zeros(shape=(self.num_units, 1))
            for j in range(self.num_units):
                w[j, 0] = float(line[j])
            self.weights.append(w)
            line = f.readline()
            b = np.zeros(shape=(1, 1))
            b[0, 0] = float(line)
            self.biases.append(b)

class Optimizer(object):
    '''
    '''

    def __init__(self, learning_rate):
        '''
        Create a Gradient Descent based optimizer with given
        learning rate.

        Other parameters can also be passed to create different types of
        optimizers.

        Hint: You can use the class members to track various states of the
        optimizer.
        '''
        self.lr = learning_rate

    def step(self, weights, biases, delta_weights, delta_biases):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
        '''
        new_w = []
        for w, dw in zip(weights, delta_weights):
            new_w.append(w - self.lr*dw)
        new_b = []
        for b, db in zip(biases, delta_biases):
            new_b.append(b - self.lr*b)
        return new_w, new_b


def loss_mse(y, y_hat):
    '''
    Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1

    Returns
    ----------
        MSE loss between y and y_hat.
    '''
    return (np.linalg.norm(y-y_hat))**2

def loss_regularization(weights, biases):
    '''
    Compute l2 regularization loss.

    Parameters
    ----------
        weights and biases of the network.

    Returns
    ----------
        l2 regularization loss 
    '''
    loss = 0
    for w in weights:
        loss += np.linalg.norm(w.flatten())**2
    for b in biases:
        loss += np.linalg.norm(b.flatten())**2
    return loss

def loss_fn(y, y_hat, weights, biases, lamda):
    '''
    Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1
        weights and biases of the network
        lamda: Regularization parameter

    Returns
    ----------
        l2 regularization loss
    '''
    return loss_mse(y, y_hat) + lamda*loss_regularization(weights, biases)

def rmse(y, y_hat):
    '''
    Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1

    Returns
    ----------
        RMSE between y and y_hat.
    '''
    m = len(y)
    return (((np.linalg.norm(y - y_hat))**2)/m)**0.5

def cross_entropy_loss(y, y_hat):
    '''
    Compute cross entropy loss

    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1

    Returns
    ----------
        cross entropy loss
    '''
    raise NotImplementedError

def train(
    net, optimizer, lamda, batch_size, max_epochs,
    train_input, train_target,
    dev_input, dev_target
):
    '''
    In this function, you will perform following steps:
        1. Run gradient descent algorithm for `max_epochs` epochs.
        2. For each bach of the training data
            1.1 Compute gradients
            1.2 Update weights and biases using step() of optimizer.
        3. Compute RMSE on dev data after running `max_epochs` epochs.

    Here we have added the code to loop over batches and perform backward pass
    for each batch in the loop.
    For this code also, you are free to heavily modify it.
    '''

    m = train_input.shape[0]

    for e in range(max_epochs):
        epoch_loss = 0.
        for i in range(0, m, batch_size):
            batch_input = train_input[i:i+batch_size]
            batch_target = train_target[i:i+batch_size]
            ## Forward prop
            pred = net(batch_input)

            # Compute gradients of loss w.r.t. weights and biases
            ## Backward prop
            dW, db = net.backward(batch_input, batch_target, lamda)

            # Get updated weights based on current weights and gradients
            ## SGD
            weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

            # Update model's weights and biases
            net.weights = weights_updated
            net.biases = biases_updated

            # Compute loss for the batch
            #print(batch_target, pred)
            batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
            epoch_loss += batch_loss

            print(e, i, rmse(batch_target, pred), batch_loss)

        #print(e, epoch_loss)

        # Write any early stopping conditions required (only for Part 2)
        # Hint: You can also compute dev_rmse here and use it in the early
        #       stopping condition.

    # After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
    dev_pred = net(dev_input)
    dev_rmse = rmse(dev_target, dev_pred)

    print('RMSE on dev data: {:.5f}'.format(dev_rmse))


def get_test_data_predictions(net, inputs):
    '''
    Perform forward pass on test data and get the final predictions that can
    be submitted on Kaggle.
    Write the final predictions to the part2.csv file.

    Parameters
    ----------
        net : trained neural network
        inputs : test input, numpy array of shape m x d

    Returns
    ----------
        predictions (optional): Predictions obtained from forward pass
                                on test data, numpy array of shape m x 1
    '''
    pred = net(inputs)
    pred = pred*200 + 1900
    pred = pred.astype(int)
    return pred

def read_data():
    '''
    Read the train, dev, and test datasets
    '''
    path = os.getcwd() + '/../regression/data'
    train_path = path + '/train.csv'
    dev_path = path + '/dev.csv'
    test_path = path + '/test.csv'

    train_data = pd.read_csv(train_path).to_numpy()
    dev_data = pd.read_csv(dev_path).to_numpy()
    test_input = pd.read_csv(test_path).to_numpy()

    train_target, train_input = train_data[:,0], train_data[:,1:]
    dev_target, dev_input = dev_data[:,0], dev_data[:,1:]
    train_target = train_target.reshape(len(train_target), 1)
    dev_target = dev_target.reshape(len(dev_target), 1)

    return train_input, train_target, dev_input, dev_target, test_input

def usage():
    print("python3 nn_1.py --<option>")
    print("List of Options:")
    print("\t\t--help: Provides help about how to run this file")
    print("\t\t--save: After training is complete the network weights and biases will be saved in trained_network.txt file")
    print("\t\t--save_file=<filename>: If you wish to provide a filename where trained network can be saved use this option. Note provide the filename with = is mandatory")
    print("\t\t--trained_file=<filename>: If a .txt file containing trained network is available then it can be added here.")

def readOptions():
    save_file = None
    trained_file = None
    if len(sys.argv) > 1:
        for option in sys.argv[1:]:
            if "help" in option:
                usage()
            if "save" in option:
                save_file = "trained_network.txt"
            if "save_file" in option:
                if '=' in option:
                    i = option.rfind('=')
                    save_file = option[i+1:]
                else:
                    save_file = "trained_network.txt"
            if "trained_file" in option:
                if '=' in option:
                    i = option.rfind('=')
                    trained_file = option[i+1:]
    return save_file, trained_file

def save_pred(pred, fname):
    with open(fname, "w") as f:
        f.write("Id,Predictions\n")
        for i in range(len(pred)):
            f.write(f"{i+1}, {pred[i, 0]}\n")

def main():

    # Hyper-parameters 
    max_epochs = 50
    batch_size = 256
    learning_rate = 1e-6
    num_layers = 1
    num_units = 64
    lamda = 0.01 # Regularization Parameter

    save_file, trained_file = readOptions()
    train_input, train_target, dev_input, dev_target, test_input = read_data()
    train_target = (train_target - 1900)/200
    dev_target = (dev_target - 1900)/200
    net = Net(num_layers, num_units)

    if trained_file is not None:
        net.set(trained_file)
    else:
        optimizer = Optimizer(learning_rate)
        train(
            net, optimizer, lamda, batch_size, max_epochs,
            train_input, train_target,
            dev_input, dev_target
        )
    if save_file is not None:
        net.save(save_file)
    pred = get_test_data_predictions(net, test_input)
    save_pred(pred, "184190001.csv")


if __name__ == '__main__':
    main()
