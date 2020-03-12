import pickle
import gzip
import numpy as np


def load_data():
    f= open('./mnistdata/data/mnist.pkl',"rb")
    training_data, validation_data, test_data = pickle.load(f,encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    train,val,test=load_data()

    training_input=[np.reshape(x,(784,1)) for x in train[0]]
    training_result=[vectorized(x) for x in train[1]]
    training_data=list(zip(training_input,training_result))

    validation_input=[np.reshape(x,(784,1)) for x in val[0]]
    validation_result=[vectorized(x) for x in val[1]]
    validation_data=list(zip(validation_input,validation_result))

    test_input=[np.reshape(x,(784,1)) for x in test[0]]
    test_result=[vectorized(x) for x in test[1]]
    test_data=list(zip(test_input,test_result))

    return (training_data, validation_data, test_data)

    

def vectorized(x):
    arr=np.zeros((10,1))
    arr[x]=1.0
    return arr



