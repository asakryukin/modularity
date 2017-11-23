import numpy as np
import random

class DataManager:

    def __init__(self):

        self.train_inputs=[]
        self.train_labels=[]

        self.test_inputs = []
        self.test_labels = []

        with open('data/mnist_train.csv','r') as file:

            for line in file:
                z = np.zeros(shape=(10, 1), dtype=np.float32)
                l=line.split(',')
                self.train_inputs.append([float(e)/255 for e in l[1:]])
                z[int(l[0])] = 1.0
                self.train_labels.append(z)

        with open('data/mnist_test.csv','r') as file:

            for line in file:
                z = np.zeros(shape=(10, 1), dtype=np.float32)
                l=line.split(',')
                self.test_inputs.append([float(e)/255 for e in l[1:]])
                z[int(l[0])] = 1.0
                self.test_labels.append(z)


        self.cursor=0
        self.max=len(self.train_labels)-1
        self.indexes=range(0,self.max+1)
        random.shuffle(self.indexes)

        self.tcursor = 0
        self.tmax = len(self.test_labels) - 1
        self.tindexes = range(0, self.tmax + 1)
        random.shuffle(self.tindexes)

    def get_batch(self,size,transofrm=None):

        if(self.cursor+size>self.max):
            random.shuffle(self.indexes)
            self.cursor=0

        batch_inputs=np.reshape([self.train_inputs[i] for i in self.indexes[self.cursor:self.cursor+size]],[size,784])

        if transofrm is not None:

            for i in xrange(0,size):
                t=np.zeros(784)
                for j in xrange(0,784):
                    t[j]=batch_inputs[i][transofrm[j]]

                batch_inputs[i]=t

        batch_labels=np.reshape([self.train_labels[i] for i in self.indexes[self.cursor:self.cursor+size]],[size,10])
        self.cursor += size



        return batch_inputs,batch_labels

    def get_test_batch(self,size,transofrm=None):

        if(self.tcursor+size>self.tmax):
            random.shuffle(self.tindexes)
            self.tcursor=0

        batch_inputs=np.reshape([self.test_inputs[i] for i in self.tindexes[self.tcursor:self.tcursor+size]],[size,784])
        if transofrm is not None:

            for i in xrange(0,size):
                t=np.zeros(784)
                for j in xrange(0,784):
                    t[j]=batch_inputs[i][transofrm[j]]

                batch_inputs[i]=t
        batch_labels=np.reshape([self.test_labels[i] for i in self.tindexes[self.tcursor:self.tcursor+size]],[size,10])
        self.tcursor += size

        return batch_inputs,batch_labels
