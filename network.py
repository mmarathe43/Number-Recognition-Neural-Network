import numpy as np
import random
import savedmodel

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

class Network(object):


    def __init__(self,sizes):
        self.layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(x,1) for x in  self.sizes[1:] ]
        self.weights=[np.random.randn(x,y) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
    
    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a=sigmoid(np.dot(w.transpose(),a)+b)
        return a
    
    def SGD(self,training_data,mini_batch_size,epochs,eta,test_data=None):
        if(test_data):
            n_test=len(test_data)
        n=len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if(test_data):
                print ("Epoch {0} : {1} => {2}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {0} complete".format(j))
        savedmodel.savemodel(self.weights,self.biases)  


    def update_mini_batch(self,mini_batch,eta):
        nabla_b=[np.zeros(y.shape) for y in self.biases]
        nabla_w=[np.zeros(x.shape) for x in self.weights]

        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w=self.backprop(x,y)
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights=[w-eta*nw/len(mini_batch) for w,nw in zip(self.weights,nabla_w)]
        self.biases=[b-eta*nb/len(mini_batch) for b,nb in zip(self.biases,nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w.transpose(),activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(activations[-2],delta.transpose())
       
        for l in range(2, self.layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1], delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = (np.dot(delta, activations[-l-1].transpose())).transpose()
        return (nabla_b, nabla_w)


    def cost_derivative(self,final_activation,y):
        return (final_activation-y)
    def evaluate(self,test_data):
        test_results=[(np.argmax(self.feedforward(x)),np.argmax(y)) for x,y in test_data]
        return sum([int(x==y) for x,y in test_results])

    def sigmoid_prime(self,z):
        return sigmoid(z)*(1-sigmoid(z))





