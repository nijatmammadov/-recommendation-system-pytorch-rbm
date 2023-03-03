# Importing The Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the datasets

movies = pd.read_csv('ml-1m/movies.dat',sep = '::',
                     header=None,engine='python',
                     encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',sep = '::',
                     header=None,engine='python',
                     encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat',sep = '::',
                     header=None,engine='python',
                     encoding = 'latin-1')


# Preparing training set and test set

training_set = pd.read_csv('ml-100k/u1.base',delimiter = '\t')
test_set = pd.read_csv('ml-100k/u1.test',delimiter = '\t')

training_set = np.array(training_set,dtype='int')
test_set = np.array(test_set,dtype='int')

nb_users = int(max(max(training_set[:,0],),max(test_set[:,0],)))
nb_movies = int(max(max(training_set[:,1],),max(test_set[:,1],)))

def convert(data):
    new_data = []
    for user_id in range(1,nb_users+1):
        id_movies = data[:,1][data[:,0]==user_id]
        id_ratings = data[:,2][data[:,0]==user_id]
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data

test_set = convert(test_set)
train_set = convert(training_set)

#Converting list to the tensor
train_set = torch.FloatTensor(train_set)
test_set = torch.FloatTensor(test_set)

train_set[train_set == 0] = -1
train_set[train_set == 1] = 0
train_set[train_set == 2] = 0
train_set[train_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


# Creating Architecture of Neural Network
"""
NOTE:
    nv - count of visible nodes
    nh - count of hidden nodes
    
    sample_h - sample of hidden nodes
    
    p_h_given_v - probability of hidden nodes given visible nodes
"""
class RBM():
    def __init__(self,nv,nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1,nh)
        self.b = torch.randn(1,nv)
        
    def sample_h(self,x):
        wx = torch.mm(x,self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self,y):
        wy = torch.mm(y,self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self,v0,vk,ph0,phk):
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()
        self.b += torch.sum((v0 - vk),0)
        self.a += torch.sum((ph0 - phk),0)
        

nv = len(train_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv,nh)

"""
Training RBM
"""
nb_epochs = 50
for epoch in range(1,nb_epochs+1):
    train_loss = 0
    s = 0.
    for id_user in range(0,nb_users - batch_size,batch_size):
        vk = train_set[id_user:id_user+batch_size]
        v0 = train_set[id_user:id_user+batch_size]
        ph0, _ = rbm.sample_h(v0)
        
        for k in range(50):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
        s += 1.
    print('epochs:',str(epoch),'loss:',str(train_loss/s))


# Testing

test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = train_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    
    if len(vt[vt>=0])>0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
    
        test_loss += torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
        s += 1.
print('loss:',str(test_loss/s))





