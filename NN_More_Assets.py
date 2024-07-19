import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import Func_More_Assets as FMA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Customize a RNN layer with double relu for multiple assets
# considering returns data to build a changed strategy weight according to price change
# v normalized
class NoTradeRegionRNN(nn.Module):

    def __init__(self, input_size, hidden_size, batch_size,dim_size):
        """Initialize params."""
        super(NoTradeRegionRNN, self).__init__()
        # read input parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dim_size = dim_size


        self.edge_coef = nn.Linear(dim_size,dim_size).to(device)


    # Forward function allows a form:
    # h_t = w_fc2*relu(w_fc1*relu(w_inp*x_t+b_inp+w_h*h_{t-1}+b_h)+b_fc1)+b_fc2+b_fc1-b_h1
    def forward(self, input, target, returns_partition, hidden):
      # create the pi_bar(merton optimal) and identity matrix
      pi_bar = (torch.tensor(target,dtype=torch.float).to(device).view(self.dim_size,self.hidden_size,self.hidden_size)*\
                torch.ones((self.dim_size,self.hidden_size,self.batch_size),dtype=torch.float).to(device)).squeeze(1)
      e_matrix = torch.eye(self.dim_size).to(device)

      def recurrence(input, hidden):
        #creating scalars, empty vectors and normalized v
        eps = 1e-5
        hidden = hidden.squeeze(1)
        hidden_temp = hidden
        judge_mat = torch.zeros([self.dim_size,self.batch_size]).to(device)
        v = torch.nn.functional.normalize(self.edge_coef.weight, p=2.0, dim=1, eps=1e-12, out = None)

        # loop once to find the fitness of each asset
        for j in range(self.dim_size):
          # create v for each asset
          vj = torch.nn.functional.normalize(self.edge_coef.weight, p=2.0, dim=1, eps=1e-12, out = None)[j,:]
          # calculate lambda for all assets
          lambda_pi_plus = (torch.abs(self.edge_coef.bias[j])*torch.ones(self.batch_size).to(device) - torch.matmul(vj,hidden-pi_bar))/(torch.matmul(vj,e_matrix[j,:]))
          lambda_pi_minus = (-torch.abs(self.edge_coef.bias[j])*torch.ones(self.batch_size).to(device) - torch.matmul(vj,hidden-pi_bar))/(torch.matmul(vj,e_matrix[j,:]))
          hidden_new = hidden + (lambda_pi_plus.view(self.batch_size,1)*e_matrix[j,:]).T*(torch.matmul(vj,hidden-pi_bar)>torch.abs(self.edge_coef.bias[j]))+\
                  (lambda_pi_minus.view(self.batch_size,1)*e_matrix[j,:]).T*(torch.matmul(vj,hidden-pi_bar)<-torch.abs(self.edge_coef.bias[j]))
          # create a matrix recording the fitness of such asset
          judge = (torch.matmul(v,hidden_new-pi_bar)<torch.abs(self.edge_coef.bias.view(self.dim_size,1))+eps) & (torch.matmul(v,hidden_new-pi_bar)>-torch.abs(self.edge_coef.bias.view(self.dim_size,1))-eps)

          judge = torch.min(judge,0).values
          judge_mat[j,:] = judge
        # create a matrix recording the assets which project to notrade region with only one projection
        judge_mat = torch.max(judge_mat,0).values
        del hidden_new
        torch.cuda.empty_cache()

        for j in range(self.dim_size):
          # create v for each asset
          vj = torch.nn.functional.normalize(self.edge_coef.weight, p=2.0, dim=1, eps=1e-12, out = None)[j,:]
          # calculate lambda for all assets
          lambda_pi_plus = (torch.abs(self.edge_coef.bias[j])*torch.ones(self.batch_size).to(device) - torch.matmul(vj,hidden_temp-pi_bar))/(torch.matmul(vj,e_matrix[j,:]))
          lambda_pi_minus = (-torch.abs(self.edge_coef.bias[j])*torch.ones(self.batch_size).to(device) - torch.matmul(vj,hidden_temp-pi_bar))/(torch.matmul(vj,e_matrix[j,:]))
          # one step projection of each asset
          hidden_temp = hidden_temp + (lambda_pi_plus.view(self.batch_size,1)*e_matrix[j,:]).T*(torch.matmul(vj,hidden_temp-pi_bar)>torch.abs(self.edge_coef.bias[j]))+\
                  (lambda_pi_minus.view(self.batch_size,1)*e_matrix[j,:]).T*(torch.matmul(vj,hidden_temp-pi_bar)<-torch.abs(self.edge_coef.bias[j]))

        del lambda_pi_plus
        del lambda_pi_minus
        torch.cuda.empty_cache()
        # start a bisection method to find the exact boundary of assets without one fitness projection
        h_in =  pi_bar
        h_out = (1-judge_mat)*hidden
        for i in range(10):
          h_m = h_in+(-h_in+h_out)/2
          judge = (torch.matmul(v,h_m-pi_bar)<=torch.abs(self.edge_coef.bias.view(self.dim_size,1))+eps) & (torch.matmul(v,h_m-pi_bar)>=-torch.abs(self.edge_coef.bias.view(self.dim_size,1))-eps)
          judge = torch.min(judge,0).values
          h_out = (~judge)*h_m+(judge)*h_out
          h_in = (judge)*h_m+(~judge)*h_in
        hidden = (judge_mat*hidden_temp+(1-judge_mat)*h_m).unsqueeze(1)
        return hidden


      output = []
      steps = range(input.size(1))
      #myret = returns
      for i in steps:
          if i ==0:
              hidden = input[:,0,:].view(self.dim_size,1,self.batch_size).to(device)
              #hidden = (torch.tensor(Markowitz_opt,dtype=torch.float).view(self.dim_size,1,1)*torch.ones((self.dim_size,1,self.batch_size),dtype=torch.float)).to(device)
          else:
              # pi_t = myrotate(pi_{t-1}*(1+r_t)/(1+sum(pi_{t-1}*r_t))) due to change of price after rebalance
              adjust_pi = hidden.view(self.dim_size,1,self.batch_size)*(1+returns_partition[:,i-1,:].view(self.dim_size,1,self.batch_size))\
                                    /(1+torch.sum(hidden.view(self.dim_size,1,self.batch_size)*returns_partition[:,i-1,:].view(self.dim_size,1,\
                                    self.batch_size),0))



              hidden = recurrence(input[:,i,:].view(self.dim_size,self.input_size,self.batch_size),  adjust_pi)

          output.append(hidden)

      output = torch.cat(output, 1)

      return output, hidden
  

# Customize a RNN layer with double relu for multiple assets
# considering returns data to build a changed strategy weight according to price change
# v not normalized
class NoTradeRegionRNN2(nn.Module):

    def __init__(self, input_size, hidden_size, batch_size,dim_size):
        """Initialize params."""
        super(NoTradeRegionRNN2, self).__init__()
        # read input parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dim_size = dim_size


        self.edge_coef = nn.Linear(dim_size,dim_size).to(device)


    # Forward function allows a form:
    # h_t = w_fc2*relu(w_fc1*relu(w_inp*x_t+b_inp+w_h*h_{t-1}+b_h)+b_fc1)+b_fc2+b_fc1-b_h1
    def forward(self, input, target, returns_partition, hidden):
      # create the pi_bar(merton optimal) and identity matrix
      pi_bar = (torch.tensor(target,dtype=torch.float).to(device).view(self.dim_size,self.hidden_size,self.hidden_size)*\
                torch.ones((self.dim_size,self.hidden_size,self.batch_size),dtype=torch.float).to(device)).squeeze(1)
      e_matrix = torch.eye(self.dim_size).to(device)

      def recurrence(input, hidden):
        #creating scalars, empty vectors and normalized v
        eps = 1e-5
        hidden = hidden.squeeze(1)
        hidden_temp = hidden
        judge_mat = torch.zeros([self.dim_size,self.batch_size]).to(device)
        v = self.edge_coef.weight

        # loop once to find the fitness of each asset
        for j in range(self.dim_size):
          # create v for each asset
          vj = self.edge_coef.weight[j,:]
          # calculate lambda for all assets
          lambda_pi_plus = (torch.abs(self.edge_coef.bias[j])*torch.ones(self.batch_size).to(device) - torch.matmul(vj,hidden-pi_bar))/(torch.matmul(vj,e_matrix[j,:]))
          lambda_pi_minus = (-torch.abs(self.edge_coef.bias[j])*torch.ones(self.batch_size).to(device) - torch.matmul(vj,hidden-pi_bar))/(torch.matmul(vj,e_matrix[j,:]))
          hidden_new = hidden + (lambda_pi_plus.view(self.batch_size,1)*e_matrix[j,:]).T*(torch.matmul(vj,hidden-pi_bar)>torch.abs(self.edge_coef.bias[j]))+\
                  (lambda_pi_minus.view(self.batch_size,1)*e_matrix[j,:]).T*(torch.matmul(vj,hidden-pi_bar)<-torch.abs(self.edge_coef.bias[j]))
          # create a matrix recording the fitness of such asset
          judge = (torch.matmul(v,hidden_new-pi_bar)<torch.abs(self.edge_coef.bias.view(self.dim_size,1))+eps) & (torch.matmul(v,hidden_new-pi_bar)>-torch.abs(self.edge_coef.bias.view(self.dim_size,1))-eps)

          judge = torch.min(judge,0).values
          judge_mat[j,:] = judge
        # create a matrix recording the assets which project to notrade region with only one projection
        judge_mat = torch.max(judge_mat,0).values
        del hidden_new
        torch.cuda.empty_cache()

        for j in range(self.dim_size):
          # create v for each asset
          vj = self.edge_coef.weight[j,:]
          # calculate lambda for all assets
          lambda_pi_plus = (torch.abs(self.edge_coef.bias[j])*torch.ones(self.batch_size).to(device) - torch.matmul(vj,hidden_temp-pi_bar))/(torch.matmul(vj,e_matrix[j,:]))
          lambda_pi_minus = (-torch.abs(self.edge_coef.bias[j])*torch.ones(self.batch_size).to(device) - torch.matmul(vj,hidden_temp-pi_bar))/(torch.matmul(vj,e_matrix[j,:]))
          # one step projection of each asset
          hidden_temp = hidden_temp + (lambda_pi_plus.view(self.batch_size,1)*e_matrix[j,:]).T*(torch.matmul(vj,hidden_temp-pi_bar)>torch.abs(self.edge_coef.bias[j]))+\
                  (lambda_pi_minus.view(self.batch_size,1)*e_matrix[j,:]).T*(torch.matmul(vj,hidden_temp-pi_bar)<-torch.abs(self.edge_coef.bias[j]))

        del lambda_pi_plus
        del lambda_pi_minus
        torch.cuda.empty_cache()
        # start a bisection method to find the exact boundary of assets without one fitness projection
        h_in =  pi_bar
        h_out = (1-judge_mat)*hidden
        for i in range(10):
          h_m = h_in+(-h_in+h_out)/2
          judge = (torch.matmul(v,h_m-pi_bar)<=torch.abs(self.edge_coef.bias.view(self.dim_size,1))+eps) & (torch.matmul(v,h_m-pi_bar)>=-torch.abs(self.edge_coef.bias.view(self.dim_size,1))-eps)
          judge = torch.min(judge,0).values
          h_out = (~judge)*h_m+(judge)*h_out
          h_in = (judge)*h_m+(~judge)*h_in
        hidden = (judge_mat*hidden_temp+(1-judge_mat)*h_m).unsqueeze(1)
        return hidden


      output = []
      steps = range(input.size(1))
      #myret = returns
      for i in steps:
          if i ==0:
              hidden = input[:,0,:].view(self.dim_size,1,self.batch_size).to(device)
              #hidden = (torch.tensor(Markowitz_opt,dtype=torch.float).view(self.dim_size,1,1)*torch.ones((self.dim_size,1,self.batch_size),dtype=torch.float)).to(device)
          else:
              # pi_t = myrotate(pi_{t-1}*(1+r_t)/(1+sum(pi_{t-1}*r_t))) due to change of price after rebalance
              adjust_pi = hidden.view(self.dim_size,1,self.batch_size)*(1+returns_partition[:,i-1,:].view(self.dim_size,1,self.batch_size))\
                                    /(1+torch.sum(hidden.view(self.dim_size,1,self.batch_size)*returns_partition[:,i-1,:].view(self.dim_size,1,\
                                    self.batch_size),0))



              hidden = recurrence(input[:,i,:].view(self.dim_size,self.input_size,self.batch_size),  adjust_pi)

          output.append(hidden)

      output = torch.cat(output, 1)

      return output, hidden




class NoTradeRegionRNN_Ellipse(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, dim_size):
        """Initialize params."""
        super(NoTradeRegionRNN_Ellipse, self).__init__()
        # Read input parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dim_size = dim_size

        self.k = int(self.dim_size * (self.dim_size - 1) / 2)
        self.num_elements = int(self.dim_size * (self.dim_size + 1) / 2)
        self.upper_triangular_elements = nn.Parameter(torch.randn(self.num_elements))

    def rotate_matrix(self):
        skew_matrix = torch.zeros(self.dim_size, self.dim_size, dtype=torch.float32).to(device)
        triu_indices = torch.triu_indices(self.dim_size, self.dim_size, offset=0).to(device)
        skew_matrix[triu_indices[0], triu_indices[1]] = self.upper_triangular_elements
        rotate = skew_matrix.T @ skew_matrix
        return rotate

    def project_to_ellipse(self, x, c, R):
        # Direction vector from center to point
        direction = x - c  # Shape: (n,)
        # The scalar to scale the direction vector such that the point lies on the ellipse boundary
        dad = direction.T @ R @ direction
        # Compute the scalar t
        t = 1.0 / torch.sqrt(dad)
        # New point on the ellipse boundary
        new_point = c + t * direction  # Shape: (n,)
        return new_point
    
    def rotate_matrix_half(self):
        skew_matrix = torch.zeros(self.dim_size, self.dim_size, dtype=torch.float32).to(device)
        triu_indices = torch.triu_indices(self.dim_size, self.dim_size, offset=0).to(device)
        skew_matrix[triu_indices[0], triu_indices[1]] = self.upper_triangular_elements
        return skew_matrix
    
    def project_to_ellipse_half(self, x, c, Q, dqn, direction):
        # Direction vector from center to point
        # direction = x - c  # Shape: (n,)
        # The scalar to scale the direction vector such that the point lies on the ellipse boundary
        # dq = torch.matmul(Q,direction)
        # Compute the scalar t
        t = 1.0 / dqn
        # New point on the ellipse boundary
        new_point = c + t * direction
        return new_point


    def forward(self, input, target, returns_partition, hidden):
        # Create the pi_bar (Merton optimal) and identity matrix
        pi_bar = (torch.tensor(target, dtype=torch.float).to(device).view(self.dim_size, self.hidden_size, self.hidden_size) *
                  torch.ones((self.dim_size, self.hidden_size, self.batch_size), dtype=torch.float).to(device)).squeeze(1)
        rotated_Q = self.rotate_matrix_half()

        def recurrence(input, hidden):
            # Creating scalars, empty vectors and normalized v
            eps = 1e-5
            hidden = hidden.squeeze(1)
            # (x-c)
            diff = hidden - pi_bar
            # x is inside or outside the ellipse
            dqn = torch.norm(torch.matmul(rotated_Q,diff),dim=0)
            judge = dqn > 1.0

            # If the point is outside the ellipse, project it to the ellipse boundary
            new_hidden = hidden.clone()
            
            t = 1.0 / dqn
            new_hidden = (pi_bar + t * diff)*judge+hidden*(~judge)
            return new_hidden.unsqueeze(1)

        output = []
        steps = range(input.size(1))
        for i in steps:
            if i == 0:
                hidden = input[:, 0, :].view(self.dim_size, 1, self.batch_size).to(device)
            else:
                adjust_pi = hidden.view(self.dim_size, 1, self.batch_size) * (1 + returns_partition[:, i - 1, :].view(self.dim_size, 1, self.batch_size)) \
                            / (1 + torch.sum(hidden.view(self.dim_size, 1, self.batch_size) * returns_partition[:, i - 1, :].view(self.dim_size, 1, self.batch_size), 0))

                hidden = recurrence(input[:, i, :].view(self.dim_size, self.input_size, self.batch_size), adjust_pi)

            output.append(hidden)

        output = torch.cat(output, 1)

        return output, hidden


class NoTradeRegionRNN_Ellipse2(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, dim_size):
        super(NoTradeRegionRNN_Ellipse2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dim_size = dim_size

        self.k = int(self.dim_size * (self.dim_size - 1) / 2)
        self.rotate_params = nn.Parameter(torch.randn(self.k))
        self.edge_params = nn.Parameter(torch.randn(self.dim_size))

    def rotate_matrix(self):
        skew_matrix = torch.zeros(self.dim_size, self.dim_size, dtype=torch.float32, device=device)
        idx = 0
        for i in range(self.dim_size):
            for j in range(i + 1, self.dim_size):
                skew_matrix[i, j] = self.rotate_params[idx]
                skew_matrix[j, i] = -self.rotate_params[idx]
                idx += 1
        rotation_matrix = torch.matrix_exp(skew_matrix)
        return rotation_matrix
    
    def positive_diag(self):
        return torch.diag(torch.exp(self.edge_params))
    
    def forward(self, input, target, returns_partition, hidden):
        rotate = self.rotate_matrix()
        diag = self.positive_diag()
        rotated_Q = rotate.T @ diag @ rotate

        # Ensure parameters are involved in computation
        output = torch.sum(rotated_Q) + torch.sum(self.edge_params)
        
        # Add some dependence on the input and hidden states for testing
        if hidden is not None:
            output += torch.sum(hidden)
        
        return output, hidden

class WealthRNN_Ellipse2(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, batch_size, seq_length, dim_size):
        super(WealthRNN_Ellipse2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.dim_size = dim_size
        self.rnn = NoTradeRegionRNN_Ellipse2(input_size, hidden_size, batch_size, dim_size).to(device)
        self.out = nn.Linear(dim_size, hidden_size, bias=False).to(device)
        self.out.weight = torch.nn.Parameter(torch.ones_like(self.out.weight)).to(device)
        self.out.weight.requires_grad = False

    def update_vec(self, value):
        self.rnn.edge_params = nn.Parameter(value).to(device)

    def forward(self, inputs, target, returns_partition, cost_partition, hidden=None):
        hidden = inputs[:, 0, :].to(device)
        output, hidden = self.rnn(inputs.float(), target, returns_partition, hidden.float())
        return output, hidden
    

# wealth process
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
        self.rnn.edge_coef.weight = torch.nn.Parameter(torch.eye(self.dim_size).to(device))
        #self.rnn.edge_coef.weight = torch.nn.functional.normalize(self.rnn.edge_coef.weight, p=2.0, dim=1, eps=1e-12, out = None)
        self.out.weight = torch.nn.Parameter(torch.ones_like(self.out.weight))
        self.out.weight.requires_grad = False

    def update_bias(self,value):
        self.rnn.edge_coef.bias = torch.nn.Parameter(value)

    def step(self, input, target, returns_partition, cost_partition, hidden=None):
        output, hidden = self.rnn(input, target, returns_partition, hidden).to(device)
        output2 = self.out.weight.view(self.dim_size,self.hidden_size,self.hidden_size)*output
        return output, output2

    def forward(self, inputs, target, returns_partition, cost_partition, hidden=None):
        hidden = inputs[:,0,:].to(device)
        output, hidden = self.rnn(inputs.float(), target, returns_partition, hidden.float())
        # output2 the overall wealth at time T
        output2 = torch.prod(FMA.cal_return(output,returns_partition,cost_partition)+1,0)
        #output2 = (torch.pow(output2,1-utility_gamma))*scaler/(1-utility_gamma)
        return  output, output2
        #return  output

# wealth function for second simpilicity(not good) 
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
        self.rnn.edge_coef.bias = torch.nn.Parameter(torch.tensor([1.0]*self.dim_size).to(device))
        self.rnn.edge_coef.bias.requires_grad = False 
        self.out.weight.requires_grad = False
        #self.rnn.edge_coef.weight = torch.nn.functional.normalize(self.rnn.edge_coef.weight, p=2.0, dim=1, eps=1e-12, out = None)
        self.out.weight = torch.nn.Parameter(torch.ones_like(self.out.weight))

    def update_weight(self,value):
        self.rnn.edge_coef.weight = torch.nn.Parameter(value)
       

    def step(self, input, target, returns_partition, cost_partition, hidden=None):
        output, hidden = self.rnn(input, target, returns_partition, hidden).to(device)
        output2 = self.out.weight.view(self.dim_size,self.hidden_size,self.hidden_size)*output
        return output, output2

    def forward(self, inputs, target, returns_partition, cost_partition, hidden=None):
        hidden = inputs[:,0,:].to(device)
        output, hidden = self.rnn(inputs.float(), target, returns_partition, hidden.float())
        # output2 the overall wealth at time T
        output2 = torch.prod(FMA.cal_return(output,returns_partition,cost_partition)+1,0)
        #output2 = (torch.pow(output2,1-utility_gamma))*scaler/(1-utility_gamma)
        return  output, output2
        #return  output

# wealth function for third simpilicity 
class WealthRNN3(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, batch_size, seq_length, dim_size):
        super(WealthRNN3, self).__init__()
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
        self.rnn.edge_coef.weight = torch.nn.Parameter(torch.eye(self.dim_size).to(device)) 
        self.rnn.edge_coef.bias.requires_grad = False 
        #self.rnn.edge_coef.weight = torch.nn.functional.normalize(self.rnn.edge_coef.weight, p=2.0, dim=1, eps=1e-12, out = None)
        self.out.weight = torch.nn.Parameter(torch.ones_like(self.out.weight))
        self.out.weight.requires_grad = False

    def update_bias(self,value):
        self.rnn.edge_coef.bias = torch.nn.Parameter(value)
        

    def step(self, input, target, returns_partition, cost_partition, hidden=None):
        output, hidden = self.rnn(input, target, returns_partition, hidden).to(device)
        output2 = self.out.weight.view(self.dim_size,self.hidden_size,self.hidden_size)*output
        return output, output2

    def forward(self, inputs, target, returns_partition, cost_partition, hidden=None):
        hidden = inputs[:,0,:].to(device)
        output, hidden = self.rnn(inputs.float(), target, returns_partition, hidden.float())
        # output2 the overall wealth at time T
        output2 = torch.prod(FMA.cal_return(output,returns_partition,cost_partition)+1,0)
        #output2 = (torch.pow(output2,1-utility_gamma))*scaler/(1-utility_gamma)
        return  output, output2
        #return  output


# wealth function for third ellipse method 
class WealthRNN_Ellipse(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, batch_size, seq_length, dim_size):
        super(WealthRNN_Ellipse, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.dim_size = dim_size
        # the rnn layer which works as out, hidden_t = f(out_(t), hidden_(t-1)), used to approximate pi^*_(t)= f(pi^*_(t-1),pi_t)
        self.rnn = NoTradeRegionRNN_Ellipse(input_size, hidden_size, batch_size, dim_size).to(device)
        self.out = nn.Linear(dim_size, hidden_size,bias=False).to(device)
        self.out.weight = torch.nn.Parameter(torch.ones_like(self.out.weight)).to(device)
        self.out.weight.requires_grad = False
        
    def initialize_diag(self, vector):
        """Initialize the diagonal values with the given vector."""
        # Create a diagonal matrix from the vector
        diagonal_matrix = torch.diag(torch.tensor(vector, dtype=torch.float32)).to(device)

        # Set the upper triangular elements of the skew_matrix
        skew_matrix = torch.zeros(self.dim_size, self.dim_size, dtype=torch.float32).to(device)
        triu_indices = torch.triu_indices(self.dim_size, self.dim_size, offset=0).to(device)
        skew_matrix[triu_indices[0], triu_indices[1]] = diagonal_matrix[triu_indices[0], triu_indices[1]]

        # Flatten the upper triangular part and assign it to the parameter
        self.rnn.upper_triangular_elements.data = skew_matrix[triu_indices[0], triu_indices[1]]
        
    def step(self, input, target, returns_partition, cost_partition, hidden=None):
        output, hidden = self.rnn(input, target, returns_partition, hidden).to(device)
        output2 = self.out.weight.view(self.dim_size,self.hidden_size,self.hidden_size)*output
        return output, output2

    def forward(self, inputs, target, returns_partition, cost_partition, hidden=None):
        hidden = inputs[:,0,:].to(device)
        output, hidden = self.rnn(inputs.float(), target, returns_partition, hidden.float())
        # output2 the overall wealth at time T
        output2 = torch.prod(FMA.cal_return(output,returns_partition,cost_partition)+1,0)
        return  output, output2
        #return  output