import gzip
import cPickle 
import numpy as np

def data_loader():
    f=gzip.open('mnist.pkl.gz','rb')
    training_data, valid_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, valid_data, test_data)
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data_wrap():
    trd,vd,testd=data_loader()
    training_inputs=[np.reshape(x,(784,1)) for x in trd[0]]
    training_results=[vectorized_result(y) for y in trd[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in vd[0]]
    validation_data = zip(validation_inputs, vd[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in testd[0]]
    test_data = zip(test_inputs, testd[1])
    return (training_data, validation_data, test_data)
