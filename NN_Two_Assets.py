import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import Func_Two_Assets as FTA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Customize a RNN layer with double relu for multiple assets
# considering returns data to build a changed strategy weight according to price change
class NoTradeRegionRNN(nn.Module):

    def __init__(self, input_size, hidden_size, batch_size,dim_size):
        """Initialize params."""
        super(NoTradeRegionRNN, self).__init__()
        # read input parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dim_size = dim_size

        # Define parameters of double relu function for activation function
        self.input_param = nn.Linear(input_size, dim_size,  hidden_size).to(device)
        self.hidden_param = nn.Linear(hidden_size,  dim_size, hidden_size).to(device)
        self.fc1_param = nn.Linear(hidden_size, dim_size,hidden_size).to(device)
        self.fc2_param = nn.Linear(hidden_size, dim_size, hidden_size).to(device)
        self.rotate_param = nn.Linear(dim_size,dim_size,bias=False)
                
    # Forward function allows a form:
    # h_t = w_fc2*relu(w_fc1*relu(w_inp*x_t+b_inp+w_h*h_{t-1}+b_h)+b_fc1)+b_fc2+b_fc1-b_h1
    def forward(self, input, target, returns_partition, hidden):
        def myrotate(input_rotate):
            input_new = input_rotate.squeeze(1)
            out = torch.matmul(self.rotate_param.weight,input_new)
            return out.unsqueeze(1)

        # a function that creates the corner of the rotated no trade region
        def corner():
            mat= np.ones([4,2,2])
            mat[0,:,:] = np.array([[-1,0],[0,-1]])
            mat[1,:,:] = np.array([[-1,0],[0,1]])
            mat[2,:,:] = np.array([[1,0],[0,1]])
            mat[3,:,:] = np.array([[1,0],[0,-1]])
            index_matrix = torch.tensor(mat,dtype = torch.float).to(device)
            res = (torch.matmul(index_matrix,self.hidden_param.bias)).T
            res = torch.matmul(self.rotate_param.weight,res).T+target
            return res

        # Create the corner and related slope of different lines
        Corner = corner()
        ac = self.rotate_param.weight[1,0]/(self.rotate_param.weight[0,0])
        bd = self.rotate_param.weight[0,1]/(self.rotate_param.weight[1,1])
        

        def lower_bound_x(x):
            ingate = (x.squeeze(1)[1]-Corner[0,1]*(bd>=0)-Corner[1,1]*(bd<0))*bd

            ingate2 = -F.relu(ingate)+torch.abs(Corner[1,0]-Corner[0,0])

            res = -F.relu(ingate2)+Corner[1,0]*(bd>=0)+Corner[0,0]*(bd<0)
            return res

        def upper_bound_x(x):
            ingate = (x.squeeze(1)[1]-Corner[3,1]*(bd>=0)-Corner[2,1]*(bd<0))*bd

            ingate2 = -F.relu(ingate)+torch.abs(Corner[2,0]-Corner[3,0])

            res = -F.relu(ingate2)+Corner[2,0]*(bd>=0)+Corner[3,0]*(bd<0)
            return res

        def lower_bound_y(x):
            ingate = (x.squeeze(1)[0]-Corner[0,0]*(ac>=0)-Corner[3,0]*(ac<0))*ac

            ingate2 = -F.relu(ingate)+torch.abs(Corner[0,1]-Corner[3,1])

            res = -F.relu(ingate2)+Corner[3,1]*(ac>=0)+Corner[0,1]*(ac<0)
            return res

        def upper_bound_y(x):
            ingate = (x.squeeze(1)[0]-Corner[1,0]*(ac>=0)-Corner[2,0]*(ac<0))*ac

            ingate2 = -F.relu(ingate)+torch.abs(Corner[1,1]-Corner[2,1])

            res = -F.relu(ingate2)+Corner[2,1]*(ac>=0)+Corner[1,1]*(ac<0)
            return res

        def lower_bound(x):
            return torch.stack((lower_bound_x(x),lower_bound_y(x))).view(self.dim_size,self.hidden_size,self.batch_size)

        def upper_bound(x):
            return torch.stack((upper_bound_x(x),upper_bound_y(x))).view(self.dim_size,self.hidden_size,self.batch_size)

        def recurrence(input, hidden):
            # w_inp*x_t+b_inp+w_h*h_{t-1}+b_h
            ingate = self.input_param.weight.view(self.dim_size,self.input_size,self.hidden_size)*input \
                    + self.hidden_param.weight.view(self.dim_size,self.hidden_size,self.hidden_size)*hidden - lower_bound(hidden)
            # w_fc1*relu(ingate)+upper-lower
            ingate2 = self.fc1_param.weight.view(self.dim_size,self.hidden_size,self.hidden_size)*F.relu(ingate)\
                    + upper_bound(hidden) - lower_bound(hidden)
            # w_fc2*relu(ingate2)+upper
            h       = self.fc2_param.weight.view(self.dim_size,self.hidden_size,self.hidden_size)*F.relu(ingate2) + upper_bound(hidden)
            return h

        output = []
        steps = range(input.size(1))
        for i in steps:
            if i ==0:
                hidden = input[:,0,:].view(self.dim_size,1,self.batch_size).to(device)
                #hidden = (torch.tensor(Markowitz_opt,dtype=torch.float).view(self.dim_size,1,1)*torch.ones((self.dim_size,1,self.batch_size),dtype=torch.float)).to(device)
            else:
                # pi_t = myrotate(pi_{t-1}*(1+r_t)/(1+sum(pi_{t-1}*r_t))) due to change of price after rebalance
                adjust_pi = hidden.view(self.dim_size,1,self.batch_size)*(1+returns_partition[:,i-1,:].view(self.dim_size,1,self.batch_size))\
                                    /(1+torch.sum(hidden.view(self.dim_size,1,self.batch_size)*returns_partition[:,i-1,:].view(self.dim_size,1,\
                                    self.batch_size),0))
                                    
                           

                hidden = recurrence(input[:,i,:].view(self.dim_size,self.input_size,self.batch_size), adjust_pi)
            
            output.append(hidden)

        output = torch.cat(output, 1)

        return output, hidden
    

# Customize a RNN layer with double relu for multiple assets
# considering returns data to build a changed strategy weight according to price change
class NoTradeRegionRNN2(nn.Module):

    def __init__(self, input_size, hidden_size, batch_size,dim_size):
        """Initialize params."""
        super(NoTradeRegionRNN2, self).__init__()
        # read input parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dim_size = dim_size

        # Define parameters of double relu function for activation function
        self.input_param = nn.Linear(input_size, dim_size,  hidden_size).to(device)
        self.hidden_param = nn.Linear(hidden_size,  dim_size, hidden_size).to(device)
        self.fc1_param = nn.Linear(hidden_size, dim_size,hidden_size).to(device)
        self.fc2_param = nn.Linear(hidden_size, dim_size, hidden_size).to(device)
        self.rotate_param = nn.Linear(dim_size,dim_size,bias=False)
                
    # Forward function allows a form:
    # h_t = w_fc2*relu(w_fc1*relu(w_inp*x_t+b_inp+w_h*h_{t-1}+b_h)+b_fc1)+b_fc2+b_fc1-b_h1
    def forward(self, input, target, returns_partition, hidden):
        def myrotate(input_rotate):
            input_new = input_rotate.squeeze(1)
            out = torch.matmul(self.rotate_param.weight,input_new)
            return out.unsqueeze(1)

        # a function that creates the corner of the rotated no trade region
        def corner():
            mat2 = np.ones([2,4])
            mat2[:,0] = np.array([-1,0])
            mat2[:,1] = np.array([0,-1])
            mat2[:,2] = np.array([1,0])
            mat2[:,3] = np.array([0,1])
            index_matrix2 = torch.tensor(mat2,dtype = torch.float).to(device)
            res = torch.matmul(self.rotate_param.weight,index_matrix2).T+target
            return res

        # Create the corner and related slope of different lines
        Corner = corner()
        ac = (self.rotate_param.weight[1,0]+self.rotate_param.weight[1,1])\
            /(self.rotate_param.weight[0,0]+self.rotate_param.weight[0,1])
        
        bd = (self.rotate_param.weight[0,0]-self.rotate_param.weight[0,1])\
            /(self.rotate_param.weight[1,0]-self.rotate_param.weight[1,1])
        

        def lower_bound_x(x):
            ingate = (x.squeeze(1)[1]-Corner[0,1]*(bd>=0)-Corner[1,1]*(bd<0))*bd

            ingate2 = -F.relu(ingate)+torch.abs(Corner[1,0]-Corner[0,0])

            res = -F.relu(ingate2)+Corner[1,0]*(bd>=0)+Corner[0,0]*(bd<0)
            return res

        def upper_bound_x(x):
            ingate = (x.squeeze(1)[1]-Corner[3,1]*(bd>=0)-Corner[2,1]*(bd<0))*bd

            ingate2 = -F.relu(ingate)+torch.abs(Corner[2,0]-Corner[3,0])

            res = -F.relu(ingate2)+Corner[2,0]*(bd>=0)+Corner[3,0]*(bd<0)
            return res

        def lower_bound_y(x):
            ingate = (x.squeeze(1)[0]-Corner[0,0]*(ac>=0)-Corner[3,0]*(ac<0))*ac

            ingate2 = -F.relu(ingate)+torch.abs(Corner[0,1]-Corner[3,1])

            res = -F.relu(ingate2)+Corner[3,1]*(ac>=0)+Corner[0,1]*(ac<0)
            return res

        def upper_bound_y(x):
            ingate = (x.squeeze(1)[0]-Corner[1,0]*(ac>=0)-Corner[2,0]*(ac<0))*ac

            ingate2 = -F.relu(ingate)+torch.abs(Corner[1,1]-Corner[2,1])

            res = -F.relu(ingate2)+Corner[2,1]*(ac>=0)+Corner[1,1]*(ac<0)
            return res

        def lower_bound(x):
            return torch.stack((lower_bound_x(x),lower_bound_y(x))).view(self.dim_size,self.hidden_size,self.batch_size)

        def upper_bound(x):
            return torch.stack((upper_bound_x(x),upper_bound_y(x))).view(self.dim_size,self.hidden_size,self.batch_size)

        def recurrence(input, hidden):
            # w_inp*x_t+b_inp+w_h*h_{t-1}+b_h
            ingate = self.input_param.weight.view(self.dim_size,self.input_size,self.hidden_size)*input \
                    + self.hidden_param.weight.view(self.dim_size,self.hidden_size,self.hidden_size)*hidden - lower_bound(hidden)
            # w_fc1*relu(ingate)+upper-lower
            ingate2 = self.fc1_param.weight.view(self.dim_size,self.hidden_size,self.hidden_size)*F.relu(ingate)\
                    + upper_bound(hidden) - lower_bound(hidden)
            # w_fc2*relu(ingate2)+upper
            h       = self.fc2_param.weight.view(self.dim_size,self.hidden_size,self.hidden_size)*F.relu(ingate2) + upper_bound(hidden)
            return h

        output = []
        steps = range(input.size(1))
        for i in steps:
            if i ==0:
                hidden = input[:,0,:].view(self.dim_size,1,self.batch_size).to(device)
                #hidden = (torch.tensor(Markowitz_opt,dtype=torch.float).view(self.dim_size,1,1)*torch.ones((self.dim_size,1,self.batch_size),dtype=torch.float)).to(device)
            else:
                # pi_t = myrotate(pi_{t-1}*(1+r_t)/(1+sum(pi_{t-1}*r_t))) due to change of price after rebalance
                adjust_pi = hidden.view(self.dim_size,1,self.batch_size)*(1+returns_partition[:,i-1,:].view(self.dim_size,1,self.batch_size))\
                                    /(1+torch.sum(hidden.view(self.dim_size,1,self.batch_size)*returns_partition[:,i-1,:].view(self.dim_size,1,\
                                    self.batch_size),0))
                                    
                           

                hidden = recurrence(input[:,i,:].view(self.dim_size,self.input_size,self.batch_size), adjust_pi)
            
            output.append(hidden)

        output = torch.cat(output, 1)

        return output, hidden
    

class WealthRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, batch_size, seq_length, dim_size):
        super(WealthRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.dim_size = dim_size
        # the rnn layer which works as out, hidden_t = f(out_(t), hidden_(t-1)), used to approximate pi^*_(t)= f(pi^*_(t-1),pi_t)
        self.rnn = NoTradeRegionRNN(input_size, hidden_size, batch_size, dim_size).to(device)
        self.out = nn.Linear(dim_size, hidden_size,bias=False).to(device)
        # initialize some bias and weight
        self.rnn.fc1_param.weight = torch.nn.Parameter(-1*torch.ones_like(self.rnn.fc1_param.weight))
        self.rnn.fc2_param.weight = torch.nn.Parameter(-1*torch.ones_like(self.rnn.fc2_param.weight))
        self.rnn.input_param.weight = torch.nn.Parameter(torch.zeros_like(self.rnn.input_param.weight))
        self.rnn.hidden_param.weight = torch.nn.Parameter(torch.ones_like(self.rnn.hidden_param.weight))
        self.out.weight = torch.nn.Parameter(*torch.ones_like(self.out.weight))
    
    def update_bias(self,value):
        self.rnn.hidden_param.bias = torch.nn.Parameter(value)
        self.rnn.fc1_param.bias = torch.nn.Parameter(2*value)
        self.rnn.fc1_param.bias.requires_grad = False
    
    def update_weight(self,value):
        
        self.rnn.rotate_param.weight = torch.nn.Parameter(value)
        
        self.rnn.input_param.weight.requires_grad = False
        self.rnn.hidden_param.weight.requires_grad = False
        self.rnn.fc1_param.weight.requires_grad = False
        self.rnn.fc2_param.weight.requires_grad = False
        self.out.weight.requires_grad = False


    def step(self, input,target, returns_partition, cost_partition, hidden=None):
        output, hidden = self.rnn(input, target, returns_partition, hidden).to(device)
        output2 = self.out.weight.view(self.dim_size,self.hidden_size,self.hidden_size)*output
        return output, output2

    def forward(self, inputs, target, returns_partition, cost_partition,hidden=None):
        hidden = self.__init__hidden().to(device)
        output, hidden = self.rnn(inputs.float(),target, returns_partition, hidden.float())
        # output2 the overall wealth at time T
        output2 = torch.prod(FTA.cal_return(output,returns_partition,cost_partition).to(device)+1,0)
        return  output, output2
        #return  output
        
    def __init__hidden(self):
        hidden = 0.0*torch.ones(self.dim_size, self.hidden_size,  self.batch_size,dtype=torch.float64).to(device)
        return hidden
    
class WealthRNN2(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, batch_size, seq_length, dim_size):
        super(WealthRNN2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.dim_size = dim_size
        # the rnn layer which works as out, hidden_t = f(out_(t), hidden_(t-1)), used to approximate pi^*_(t)= f(pi^*_(t-1),pi_t)
        self.rnn = NoTradeRegionRNN2(input_size, hidden_size, batch_size, dim_size).to(device)
        self.out = nn.Linear(dim_size, hidden_size,bias=False).to(device)
        # initialize some bias and weight
        self.rnn.fc1_param.weight = torch.nn.Parameter(-1*torch.ones_like(self.rnn.fc1_param.weight))
        self.rnn.fc2_param.weight = torch.nn.Parameter(-1*torch.ones_like(self.rnn.fc2_param.weight))
        self.rnn.input_param.weight = torch.nn.Parameter(torch.zeros_like(self.rnn.input_param.weight))
        self.rnn.hidden_param.weight = torch.nn.Parameter(torch.ones_like(self.rnn.hidden_param.weight))
        self.out.weight = torch.nn.Parameter(*torch.ones_like(self.out.weight))
    
    def update_bias(self,value):
        self.rnn.hidden_param.bias = torch.nn.Parameter(value)
        self.rnn.fc1_param.bias = torch.nn.Parameter(2*value)
        self.rnn.fc1_param.bias.requires_grad = False
    
    def update_weight(self,value):
        
        self.rnn.rotate_param.weight = torch.nn.Parameter(value)
        
        self.rnn.input_param.weight.requires_grad = False
        self.rnn.hidden_param.weight.requires_grad = False
        self.rnn.fc1_param.weight.requires_grad = False
        self.rnn.fc2_param.weight.requires_grad = False
        self.out.weight.requires_grad = False


    def step(self, input,target, returns_partition, cost_partition, hidden=None):
        output, hidden = self.rnn(input, target, returns_partition, hidden).to(device)
        output2 = self.out.weight.view(self.dim_size,self.hidden_size,self.hidden_size)*output
        return output, output2

    def forward(self, inputs, target, returns_partition, cost_partition,hidden=None):
        hidden = self.__init__hidden().to(device)
        output, hidden = self.rnn(inputs.float(),target, returns_partition, hidden.float())
        # output2 the overall wealth at time T
        output2 = torch.prod(FTA.cal_return(output,returns_partition,cost_partition).to(device)+1,0)
        return  output, output2
        #return  output
        
    def __init__hidden(self):
        hidden = 0.0*torch.ones(self.dim_size, self.hidden_size,  self.batch_size,dtype=torch.float64).to(device)
        return hidden