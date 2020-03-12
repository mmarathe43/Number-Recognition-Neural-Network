import pickle

def savemodel(weights,biases):
    f=open('./saveddata.pkl',"wb")
    pickle.dump(list(zip(weights,biases)),f)
    f.close()

def getmodel():
    f=open('./saveddata.pkl',"rb")
    array= pickle.load(f,encoding='latin1')
    f.close()
    return (array)
