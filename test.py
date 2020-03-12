import numpy as np
import matplotlib.pyplot as plt
import mnist_loader
training_data, validation_data, test_data =mnist_loader.load_data_wrapper()

import savedmodel
array = savedmodel.getmodel()

def feedforward(a,array):
    for w,b in array:
        a=sigmoid(np.dot(w.transpose(),a)+b)
    return a

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


image=test_data[550]
# plt.imshow(image,cmap=plt.cm.Greys)
# plt.show()
# print(image)
def show_result(activation):
    # result= feedforward(activation[0],weights,biases)
    # print(len(result))
    print("prediction :-")
    print(np.argmax(feedforward(activation[0],array)))
    print("actual :-")
    print(np.argmax(activation[1]))
    plt.imshow(activation[0])
    plt.show()

show_result(image)