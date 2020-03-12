import data_preprocess
training_data, validation_data, test_data =data_preprocess.load_data_wrapper()


import network
net= network.Network([784,30,10])

net.SGD(training_data,10,1,3,test_data)
