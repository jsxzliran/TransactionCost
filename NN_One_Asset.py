import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import math, random
import Func_One_Asset as FOA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Customize a RNN layer with double relu
# considering returns data to build a changed strategy weight according to price change
class NoTradeRegionRNN(nn.Module):

    def __init__(self, input_size, hidden_size, batch_size):
        """Initialize params."""
        super(NoTradeRegionRNN, self).__init__()
        # read input parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.input_param = nn.Linear(input_size,  hidden_size,bias = False).to(device)
        self.hidden_param = nn.Linear(hidden_size,  hidden_size).to(device)
        self.fc1_param = nn.Linear(hidden_size,hidden_size).to(device)
        self.fc2_param = nn.Linear(hidden_size,hidden_size,bias = False).to(device)
        
    # Forward function allows a form:
    # h_t = w_fc2*relu(w_fc1*relu(w_inp*x_t+b_inp+w_h*h_{t-1}+b_h)+b_fc1)+b_fc2+b_fc1-b_h1
    def forward(self, input, target, returns, hidden):
        pi_bar = torch.tensor(target,dtype=torch.float).to(device)
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            
            ingate = self.input_param(input) + self.hidden_param.weight*hidden-(pi_bar-self.hidden_param.bias)
            ingate2 = self.fc1_param.weight*F.relu(ingate)+2*self.hidden_param.bias
            h = self.fc2_param.weight*F.relu(ingate2)+pi_bar+self.hidden_param.bias
            return h


        # Loop to formulate the rnn
        output = []
        steps = range(input.size(0))
        myret = returns.view(input.size(0)-1,self.batch_size,self.hidden_size)
        for i in steps:
            if i ==0:
                hidden = input[0]*torch.ones(self.hidden_size,self.batch_size,self.hidden_size).to(device)
            else:
            # pi_t = pi_{t-1}*(1+r_t)/(1+pi_{t-1}*r_t) due to change of price after rebalance
                hidden = recurrence(input[i], hidden*(1+myret[i-1])/(1+hidden*myret[i-1]))
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), self.batch_size, self.hidden_size)

        return output, hidden
    

class WealthRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, batch_size, seq_length):
        super(WealthRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.seq_length = seq_length
        # the rnn layer which works as out, hidden_t = f(out_(t), hidden_(t-1)), used to approximate pi^*_(t)= f(pi^*_(t-1),pi_t)
        self.rnn = NoTradeRegionRNN(input_size, hidden_size, batch_size).to(device)
        self.out = nn.Linear(hidden_size, hidden_size,bias=False).to(device)
        # initialize some bias and weight
        self.rnn.input_param.weight = torch.nn.Parameter(torch.zeros(1,1))
        self.rnn.hidden_param.weight = torch.nn.Parameter(torch.ones(1,1))
        self.rnn.hidden_param.bias = torch.nn.Parameter(0*torch.ones(1,1))
        self.rnn.fc1_param.bias = torch.nn.Parameter(0*torch.ones(1,1))
        self.rnn.fc1_param.weight = torch.nn.Parameter(-1*torch.ones(1,1))
        self.rnn.fc2_param.weight = torch.nn.Parameter(-1*torch.ones(1,1))
        self.out.weight = torch.nn.Parameter(torch.ones(hidden_size,hidden_size))

    def update_bias(self,value):
        self.rnn.hidden_param.bias = torch.nn.Parameter(value*torch.ones(1,1))
        self.out.weight.requires_grad = False
        self.rnn.input_param.weight.requires_grad = False
        self.rnn.hidden_param.weight.requires_grad = False
        self.rnn.fc1_param.weight.requires_grad = False
        self.rnn.fc2_param.weight.requires_grad = False
    
    def step(self, input, target, returns, cost, hidden=None):
        output, hidden = self.rnn(input, target, returns, hidden).to(device)
        output2 = self.out(output)
        return output, output2

    def forward(self, inputs, target, returns, cost,hidden=None):
    #    hidden = self.__init__hidden().to(device)
        hidden = inputs[0]*torch.ones(self.n_layers, self.batch_size, self.hidden_size, dtype=torch.float64).to(device)
        output, hidden = self.rnn(inputs.float(), target, returns,hidden.float())
        # output_temp the overall wealth at time T for importance sampling
        output2 = torch.prod(FOA.cal_return(output.float().view(self.seq_length,self.batch_size),returns,cost).to(device)+1,0)
        return  output, output2
        #return  output

#    def __init__hidden(self):
#       hidden = inputs[0]*torch.ones(self.n_layers, self.batch_size, self.hidden_size, dtype=torch.float64).to(device)
#       return hidden