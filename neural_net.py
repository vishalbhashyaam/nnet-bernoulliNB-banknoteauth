import numpy as np
import math
import scipy as sp
import pandas as pd
import data_cleaning
from sklearn.metrics import log_loss

def error_calc(ypred,ytrue):
        sq_err = 0
        for i in range(len(ypred)):
            sq_err += (ytrue[i]*np.log(ypred[i])) + (1-ytrue[i])*np.log(1-ypred[i]) 
        
        
            return sq_err/len(ypred)

class net: 
    def __init__(self,input,layer,target):
        self.input = input
        self.layer = layer 
        self.bias = 0
        self.target = target 

   
        
    def initial_weights(self,input): 
        np.random.seed(7)
        w1,w2,w3,w4 = [],[],[],[]
        
        
        for i in range(0,4): 
            w1.append(np.random.uniform(0,0.5))
            w2.append(np.random.uniform(0,0.5))
    
        w3.append(np.random.uniform(0,0.5))
        w4.append(np.random.uniform(0,0.5))
        
        return w1,w2,w3,w4
        
    def activation (self,x):

        sigmoid = 1 / (1-np.exp(-x))
        return sigmoid

    def feed_forward(self):
        w1,w2,w3,w4 = net.initial_weights(self, input=self.input)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w3 = w4

        category = False
        out_h1 = 0
        out_h2 =0  
        pred_cat= np.array([],dtype=np.int64)
        y = 0 
        y_out = []
        for k in range (0,len(self.input)):
            for i in range (0,1):
                out_h1 = w1[i]* self.input[k][i] + w1[i+1]* self.input[k][i+1] + w1[i+2]* self.input[k][i+2] + w1[i+3]* self.input[k][i+3] + self.bias

                out_h2 = w2[i]* self.input[k][i] + w2[i+1]* self.input[k][i+1] + w2[i+2]* self.input[k][i+2] + w2[i+3]* self.input[k][i+3] + self.bias

                
        
                hid_2 =(out_h1*w3[0]) +(out_h2*w4[0])

                y_out= np.append(y_out,hid_2)

                if (y_out[k]>=  0.5):
                    pred_cat = np.append(pred_cat,1)

                else:
                    pred_cat = np.append(pred_cat,0)

            out_h1 = 0
            out_h2 = 0

            
        
        return y_out


if __name__ == "__main__":


    data = data_cleaning.rand_data(data_cleaning.read_data())

    inputs = data_cleaning.input_array(data = data)

    target = data_cleaning.target_array(data= data)

    layers = 1


    nnet = net(input= inputs , layer= layers, target= target)

    target = target.transpose().reshape(1372)

    # Prediction for a step of the process 
    prediction = nnet.feed_forward()
    
    #  error with log loss from scratch 

   
    print(error_calc(prediction,target))
    
    
    print("Squared error for network = ",log_loss(target,prediction))

  