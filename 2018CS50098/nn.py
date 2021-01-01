# from sklearn.neural_network import MLPClassifier
import warnings
import numpy as np
import sys
import math

class Layer:

    def __init__ (self,size,input):
        self.size = size
        self.W = np.random.randn(size,input+1)*np.sqrt(1/(input+1))
        self.change = np.array([[]])
        self.output = np.array([[]])
        self.delta = np.array([[]])

class NeuralNetwork:

    def __init__ (self,batch_size,n,layers_size,r,cmd):
        self.batch_size = batch_size
        self.n = n
        self.r = r
        self.cmd = cmd
        self.layers_size = layers_size
        self.layers = []
        if len(layers_size) > 0:
            layer = Layer(layers_size[0],n)
            self.layers.append(layer)
        else:
            layer = Layer(r,n)
            self.layers.append(layer)
        for i in range(1,len(layers_size)):
            layer = Layer(layers_size[i],layers_size[i-1])
            self.layers.append(layer)
        if len(layers_size) > 0:
            layer = Layer(r,layers_size[-1])
            self.layers.append(layer)

    def activation(self, X):
        if self.cmd == "relu": return np.where(X>0, X, 0)
        else: return self.sigmoid(X)
    def derivative(self, X):
        if self.cmd == "relu": return np.where(X>0, 1, 0)
        else: return self.del_sigmoid(X)
    def sigmoid(self, X): return 1/(1+np.exp(-X))
    def del_sigmoid(self, X): return X*(1-X)

    def train(self,X_train,y_train):
        # parameters #
        lr = 0.1     #
        alpha = 1e-4 #
        k = 10       #
        epoch_ = 500 #
        # parameters #
        prev_cost = -10
        count = 0
        epochs = 0
        data = np.column_stack((X_train,y_train))
        np.random.shuffle(data)
        X_train = data[:,:-10]
        y_train = data[:,-10:]
        while True:
            cost = 0
            for b in range(math.ceil(X_train.shape[0]/self.batch_size)):
                # forward propagation
                if len(self.layers) > 1: self.layers[0].output = self.activation(self.layers[0].W @ np.row_stack((X_train[b*self.batch_size:(b+1)*self.batch_size].T,np.ones(X_train[b*self.batch_size:(b+1)*self.batch_size].T.shape[1]))))
                else: self.layers[0].output = self.sigmoid(self.layers[0].W @ np.row_stack((X_train[b*self.batch_size:(b+1)*self.batch_size].T,np.ones(X_train[b*self.batch_size:(b+1)*self.batch_size].T.shape[1]))))
                for i in range(1,len(self.layers)-1):
                    self.layers[i].output = self.activation(self.layers[i].W @ np.row_stack((self.layers[i-1].output,np.ones(self.layers[i-1].output.shape[1]))))
                if len(self.layers) > 1: self.layers[-1].output = self.sigmoid(self.layers[-1].W @ np.row_stack((self.layers[-2].output,np.ones(self.layers[-2].output.shape[1]))))
                # loss calculation
                cost += np.sum((y_train[b*self.batch_size:(b+1)*self.batch_size].T - self.layers[-1].output) ** 2) / (2 * X_train.shape[0])
                # back propagation
                self.layers[-1].delta = self.del_sigmoid(self.layers[-1].output) * (self.layers[-1].output - y_train[b*self.batch_size:(b+1)*self.batch_size].T)
                for i in range(len(self.layers)-1,0,-1):
                    self.layers[i].change = (lr * (self.layers[i].delta @ np.column_stack((self.layers[i-1].output.T,np.ones(self.layers[i-1].output.shape[1])))) / self.batch_size)
                    self.layers[i-1].delta = self.derivative(self.layers[i-1].output) * (self.layers[i].W.T[0:-1] @ self.layers[i].delta)
                self.layers[0].change = (lr * (self.layers[0].delta @ np.column_stack((X_train[b*self.batch_size:(b+1)*self.batch_size],np.ones(X_train[b*self.batch_size:(b+1)*self.batch_size].T.shape[1])))) / self.batch_size)
                # update the parameters
                for i in range(len(self.layers)):
                    self.layers[i].W -= self.layers[i].change
            epochs += 1
            # print(f"Epochs = {epochs} and Loss = {round(cost,6)}")
            # stopping criteria
            if epochs == epoch_: break
            if abs(cost-prev_cost) < alpha:
                count += 1
                if count > k: break
            else: count = 0
            prev_cost = cost
            lr = 0.5 / math.sqrt(epochs + 1)

    def test(self,X_test,y_test):
        accurate = []
        cost = 0
        for b in range(math.ceil(X_test.shape[0]/self.batch_size)):
            # forward propagation
            if len(self.layers) > 1: self.layers[0].output = self.activation(self.layers[0].W @ np.row_stack((X_test[b*self.batch_size:(b+1)*self.batch_size].T,np.ones(X_test[b*self.batch_size:(b+1)*self.batch_size].T.shape[1]))))
            else: self.layers[0].output = self.sigmoid(self.layers[0].W @ np.row_stack((X_test[b*self.batch_size:(b+1)*self.batch_size].T,np.ones(X_test[b*self.batch_size:(b+1)*self.batch_size].T.shape[1]))))
            for i in range(1,len(self.layers)-1):
                self.layers[i].output = self.activation(self.layers[i].W @ np.row_stack((self.layers[i-1].output,np.ones(self.layers[i-1].output.shape[1]))))
            if len(self.layers) > 1: self.layers[-1].output = self.sigmoid(self.layers[-1].W @ np.row_stack((self.layers[-2].output,np.ones(self.layers[-2].output.shape[1]))))
            # loss calculation
            cost += np.sum((y_test[b*self.batch_size:(b+1)*self.batch_size].T - self.layers[-1].output) ** 2) / (2 * X_test.shape[0])
            # accuracy calculation
            accurate.extend([1 for i in range(len(self.layers[-1].output.T)) if np.argmax(self.layers[-1].output.T[i]) == np.argmax(y_test[b*self.batch_size:(b+1)*self.batch_size][i])])
        print(f"Accuracy = {round(100*sum(accurate)/y_test.shape[0],2)}% and Loss = {round(cost,6)}")

    def predict(self,X):
        y = []
        for b in range(math.ceil(X.shape[0]/self.batch_size)):
            # forward propagation
            if len(self.layers) > 1: self.layers[0].output = self.activation(self.layers[0].W @ np.row_stack((X[b*self.batch_size:(b+1)*self.batch_size].T,np.ones(X[b*self.batch_size:(b+1)*self.batch_size].T.shape[1]))))
            else: self.layers[0].output = self.sigmoid(self.layers[0].W @ np.row_stack((X[b*self.batch_size:(b+1)*self.batch_size].T,np.ones(X[b*self.batch_size:(b+1)*self.batch_size].T.shape[1]))))
            for i in range(1,len(self.layers)-1):
                self.layers[i].output = self.activation(self.layers[i].W @ np.row_stack((self.layers[i-1].output,np.ones(self.layers[i-1].output.shape[1]))))
            if len(self.layers) > 1: self.layers[-1].output = self.sigmoid(self.layers[-1].W @ np.row_stack((self.layers[-2].output,np.ones(self.layers[-2].output.shape[1]))))
            # class prediction
            y.extend([np.argmax(label) for label in self.layers[-1].output.T])
        return y

def one_hot(y):
    return np.array([[1 if i==y[j][0] else 0 for i in range(10)] for j in range(y.shape[0])])

X_train=np.load(sys.argv[1])
X_train=X_train.reshape(X_train.shape[0],-1)
X_train=X_train/255
y_train=np.load(sys.argv[2])
y_train=np.outer(y_train,np.ones(1))
y_train=one_hot(y_train)

X_test=np.load(sys.argv[3])
X_test=X_test.reshape(X_test.shape[0],-1)
X_test=X_test/255
# y_test=np.load("y_test.npy")
# y_test=np.outer(y_test,np.ones(1))
# y_test=one_hot(y_test)

nn = NeuralNetwork(int(sys.argv[5]), X_train.shape[1], [int(i) for i in sys.argv[6].split()], 10, sys.argv[7])
nn.train(X_train,y_train)
np.savetxt(sys.argv[4], nn.predict(X_test), fmt="%d", delimiter="\n")
# nn.test(X_train, y_train)
# nn.test(X_test,y_test)

# warnings.filterwarnings("ignore")
# classifier = MLPClassifier(hidden_layer_sizes=(100,100),activation='relu',solver='sgd',max_iter=500)
# classifier.fit(X_train, y_train)
# print(f"Accuracy on Train Data = {round(100 * classifier.score(X_train,y_train), 2)}% using MLPClassifier")
# print(f"Accuracy on Test Data = {round(100 * classifier.score(X_test,y_test), 2)}% using MLPClassifier")
