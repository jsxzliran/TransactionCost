import numpy as np
import torch
import torch.nn as nn
import scipy.stats as stats
from torch import optim
import Data_generator_multiple as dg
import Utility_Loss as UL
import NN_Two_Assets as NTA
import Plot_More_Assets as PMA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# calculate return
def cal_return(strat_partition, return_partition, cost_partition, n_iterations=5):
    r = torch.sum(strat_partition[:, :-1, :] * return_partition, 0)
    r0 = torch.sum(strat_partition[:, :-1, :] * return_partition, 0)

    for _ in range(n_iterations):
        r = r0 - torch.sum(cost_partition * abs((r + 1) * strat_partition[:, 1:, :] - (return_partition + 1) * strat_partition[:, :-1, :]), axis=0)

    return r

class Cal_return(nn.Module):
    def __init__(self):
        super(Cal_return,self).__init__()

    def forward(self, strat_partition, return_partition, cost_partition):
        return cal_return(strat_partition, return_partition, cost_partition)
    
# create a portfolio
def make_portfolio(i,seed,mu,cov,s0,total_path,num_stocks,seq_length,T,trade_cost,utility_gamma):
    all_cost = np.ones([num_stocks,seq_length-1,total_path])*(trade_cost[i,:].reshape([num_stocks,1,1]))
    # Create a stock simulation with prices, returns
    # seed, mu, sigma, S0, paths, steps, T
    # Define the Merton_optimal strategy
    Merton_opt = np.matmul(np.linalg.inv(cov),mu)/utility_gamma
    Merton_opt_tensor =  torch.tensor(Merton_opt,dtype = torch.float).to(device)
    
    
    stock = dg.ManyStocks(seed,num_stocks,mu.T,s0,cov,total_path,seq_length-1,T)
    returns = torch.tensor(stock.Returns(),dtype=torch.float).to(device)
    # Create a default strategy as initial input, better use the optimal strategy without cost
    strategy = (torch.tensor(Merton_opt,dtype=torch.float).view(num_stocks,1,1)*torch.ones((num_stocks,seq_length,total_path),dtype=torch.float)).to(device)
    # Create a trading cost
    cost =  torch.tensor(all_cost,dtype=torch.float).to(device)

    delta = torch.pow(1.5/utility_gamma*(strategy*strategy*(1-strategy)*(1-strategy))[:,0,0]*cost[:,0,0],1/3)
    delta_tensor = torch.tensor(delta, dtype = torch.float).to(device)
    return returns, strategy, cost, Merton_opt, Merton_opt_tensor, delta_tensor

# Module for importance sampling
def importance_sampling(is_importance,seed,mu,cov,s0,num_stocks, total_path,seq_length,utility_gamma,T):
    """
    Simulates stock returns under a different probability measure using importance sampling.

    Parameters:
    - is_importance (bool): Flag to determine if importance sampling should be used.
    - seed (int): Random seed for reproducibility.
    - mu (float): Drift rate of the stock under the original measure.
    - sigma (float): Volatility of the stock.
    - s0 (float): Initial stock price.
    - npaths (int): Number of simulation paths.
    - seq_length (int): The sequence length of the simulation.
    - gamma (float): Risk aversion parameter.
    - T (float): Time horizon for the simulation.
    - device (str or torch.device): The device on which to perform the computations.

    Returns:
    Tuple of (mu_importance, returns, scaler) under the Q measure.
    """
    if not is_importance:
        stock = dg.ManyStocks(seed,num_stocks,mu.T,s0,cov,total_path,seq_length-1,T)
        returns = torch.tensor(stock.Returns(),dtype=torch.float).to(device)
        scaler = 1
        
    else:
        vol = np.linalg.cholesky(cov)
        # define the h
        h = (1-utility_gamma)/utility_gamma*np.matmul(vol,np.matmul(np.linalg.inv(cov),mu.reshape([num_stocks,1])))
        # define the new drift under Q measure
        mu_importance = np.matmul(vol,h).reshape([num_stocks])+mu
        # Simulate stock under Q measure
        returns = torch.tensor(dg.ManyStocks(seed,num_stocks,mu_importance.T,s0,cov,total_path,seq_length-1,T).Returns(),dtype=torch.float).to(device)
        # Extract Brownian Motion under Q measure
        BM_last = np.cumsum(dg.ManyStocks(seed,num_stocks,mu_importance.T,s0,cov,total_path,seq_length-1,T).BM_Ind(),1)[:,-1,:]
        # The scaler
        scaler_comp1 = -1/2*np.matmul(h.T,h)*T
        scaler_comp2 = -h.reshape([num_stocks,1])*BM_last
        scaler = torch.tensor(np.exp(np.sum(scaler_comp2,0)+scaler_comp1),dtype=torch.float).to(device).squeeze(0)
    return returns, scaler

# set up an initial rotate matrix
def initial_rotate_matrix(cov,delta_tensor, num_stocks):
    if num_stocks!=2:
        return("error:only for 2 assets")
    else:
        matrix  = torch.zeros(num_stocks,num_stocks).to(device)
        matrix[:,0]=delta_tensor
        matrix[:,1]=delta_tensor
        matrix[1,1] = -matrix[1,1]
        # if cov[0,1]==0:
        #     matrix[:,0]=delta_tensor
        #     matrix[:,1]=delta_tensor
        #     matrix[1,1] = -matrix[1,1]
        # elif cov[0,1]>0:
        #     matrix[:,0]=0.95*delta_tensor
        #     matrix[:,1]=1.05*delta_tensor
        #     matrix[1,1] = -matrix[1,1]
        # else:
        #     matrix[:,0]=1.05*delta_tensor
        #     matrix[:,1]=0.95*delta_tensor
        #     matrix[1,1] = -matrix[1,1]
        return matrix


def simpleNTR(input, returns, upper, lower):
    output = []
    steps = range(input.size(1))
    for i in steps:
        if i == 0:
            hidden = input[:, 0, :].view(input.size(0), 1, input.size(2)).to(device)
        else:
            adjust_pi = hidden.view(input.size(0), 1, input.size(2)) * (1 + returns[:, i - 1, :].view(input.size(0), 1, input.size(2))) \
                        / (1 + torch.sum(hidden.view(input.size(0), 1, input.size(2)) * returns[:, i - 1, :].view(input.size(0), 1, input.size(2)), 0))
            # Apply the simple no trade region strategy
            hidden = torch.where(adjust_pi < lower, lower, torch.where(adjust_pi > upper, upper, adjust_pi))
        output.append(hidden)
    output = torch.cat(output, 1)
    return output

def cal_esr(strategy, returns, cost, utility_gamma, T, scaler = 1):
  x = torch.prod(cal_return(strategy,returns,cost)+1,0)
  esr = torch.log(torch.pow(torch.mean(torch.pow(x,1-utility_gamma)*scaler),1/(1-utility_gamma)))/T
  #I = np.log(np.power(confidence_interval((torch.pow(x,1-utility_gamma)*scaler).detach().cpu().numpy()),1/(1-utility_gamma)))/T
  std = torch.std(torch.log(torch.pow((torch.pow(x,1-utility_gamma)*scaler),1/(1-utility_gamma)))/T)
  return esr,std

def make_model(input_size, hidden_size, n_layers, num_stocks, npaths, seq_length, delta_tensor, initial_rotate_matrix, utility_gamma, learning_rate):
    batch_size = npaths
    dim_size = num_stocks
    model = NTA.WealthRNN(input_size, hidden_size, n_layers, batch_size, seq_length,dim_size).to(device)
    criterion = UL.PowerUtilityLoss(utility_gamma)
    model.update_weight(initial_rotate_matrix)
    model.update_bias(delta_tensor)
    model = model.to(torch.float32)
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    return model, criterion, optimizer

def make_model2(input_size, hidden_size, n_layers, num_stocks, npaths, seq_length, delta_tensor, initial_rotate_matrix, utility_gamma, learning_rate):
    batch_size = npaths
    dim_size = num_stocks
    model = NTA.WealthRNN2(input_size, hidden_size, n_layers, batch_size, seq_length,dim_size).to(device)
    criterion = UL.PowerUtilityLoss(utility_gamma)
    model.update_weight(initial_rotate_matrix)
    model.update_bias(delta_tensor)
    model = model.to(torch.float32)
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    return model, criterion, optimizer

def train_model(strategy, target, returns, cost, scaler, model, criterion, optimizer,n_epochs,is_batch,n_partition,npaths,utility_gamma,T):
    losses = np.zeros(n_epochs+1)
    loss = 0
    # train with small batches
    if is_batch:
        for epoch in range(n_epochs+1):
            for batches in range(n_partition):
                _, outputs = model(strategy.to(device)[:,:,batches*npaths:(batches+1)*npaths],target, returns[:,:,batches*npaths:(batches+1)*npaths], cost[:,:,batches*npaths:(batches+1)*npaths], None)

                loss = criterion(outputs,scaler[batches*npaths:(batches+1)*npaths])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #scheduler.step()
            losses[epoch] += loss
    # train with full batch:
    else:
        for epoch in range(n_epochs+1):
            _, outputs = model(strategy.to(device),target, returns, cost, None)
            loss = criterion(outputs,scaler)
            #loss = criterion(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
            losses[epoch] += loss
    PMA.plot_loss_esr(n_epochs, losses,utility_gamma,T)
    return model, losses

# add a function to calculate confidence interval
def confidence_interval(data, confidence_level=0.95):
    """
    Calculate the confidence interval for the mean of a given set of numbers
    for any specified confidence level.

    :param data: A list-like sequence of numerical values.
    :param confidence_level: The desired confidence level for the interval (e.g., 0.95 for 95%).
    :return: A tuple containing the lower and upper bounds of the specified confidence interval.
    """
    # Convert the data to a numpy array for easier mathematical operations
    data = np.array(data)

    # Calculate the mean
    mean = np.mean(data)

    # Calculate the standard error of the mean (SEM)
    sem = stats.sem(data)

    # Determine the z-score for the specified confidence level
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    # Calculate the margin of error
    margin_of_error = z_score * sem

    # Calculate the confidence interval
    confidence_lower = mean - margin_of_error
    confidence_upper = mean + margin_of_error
    return (confidence_lower, confidence_upper)



def ESR(input_size, hidden_size, n_layers,num_stocks, seq_length, npaths,\
        model_state_dict,  strategy, returns, cost, utility_gamma, T, scaler = 1, confidence = 0.95):
    
    batch_size = npaths
    seq_length = seq_length
    dim_size = num_stocks
    # create a simple no trade region
    lower = strategy[:,0,:].view(num_stocks,1,npaths)-torch.pow(1.5/utility_gamma*(strategy*strategy*(1-strategy)*(1-strategy))[:,0,:].view(num_stocks,1,npaths)*cost[:,0,:].view(num_stocks,1,npaths),1/3)
    upper = strategy[:,0,:].view(num_stocks,1,npaths)+torch.pow(1.5/utility_gamma*(strategy*strategy*(1-strategy)*(1-strategy))[:,0,:].view(num_stocks,1,npaths)*cost[:,0,:].view(num_stocks,1,npaths),1/3)
    stra_NTR_the = simpleNTR(strategy, returns, upper, lower)
    # create a new model
    model3 = NTA.WealthRNN(input_size, hidden_size, n_layers, batch_size, seq_length,dim_size).to(device)
    # load saved data
    model3.load_state_dict(model_state_dict)
    # outputs of testing data
    _, outputs = model3(strategy.to(device),strategy[:,0,0], returns, cost, None)
    # condidence interval
    CI = 1/T*np.log(np.power(confidence_interval(np.power(outputs.detach().cpu().numpy(),1-utility_gamma),confidence),1/(1-utility_gamma)))

    return cal_esr(strategy,returns,cost,utility_gamma, T),cal_esr(stra_NTR_the,returns,cost,utility_gamma, T),cal_esr(_,returns,cost,utility_gamma, T),CI

def ESR_special(mu,cov,utility_gamma,trade_cost):
    rho = np.round(cov[0,1]/(np.sqrt(cov[0,0]*cov[1,1])),3)
    Merton_opt = np.matmul(np.linalg.inv(cov),mu)/utility_gamma
    sigma_I = np.sqrt(cov[1,1]*(1-rho*rho))
    beta2 = np.sqrt(cov[1,1]*rho*rho/cov[0,0])
    alpha = mu[1]-beta2*mu[0]
    Inside = np.power(beta2-mu[0]/utility_gamma/cov[0,0],2)*cov[0,0]+np.power((1-Merton_opt[1])*sigma_I,2)
    ESR_Theo =1/2/utility_gamma*(mu[0]*mu[0]/cov[0,0]+alpha*alpha/sigma_I/sigma_I)-\
      np.power(3*Merton_opt[1]*Merton_opt[1]*sigma_I/8/np.sqrt(2*utility_gamma)*Inside*2*trade_cost[1],2/3)
    return ESR_Theo


def ESR2(input_size, hidden_size, n_layers,num_stocks, seq_length, npaths,\
        model_state_dict,  strategy, returns, cost, utility_gamma, T, scaler = 1, confidence = 0.95):
    
    batch_size = npaths
    seq_length = seq_length
    dim_size = num_stocks
    # create a simple no trade region
    lower = strategy[:,0,:].view(num_stocks,1,npaths)-torch.pow(1.5/utility_gamma*(strategy*strategy*(1-strategy)*(1-strategy))[:,0,:].view(num_stocks,1,npaths)*cost[:,0,:].view(num_stocks,1,npaths),1/3)
    upper = strategy[:,0,:].view(num_stocks,1,npaths)+torch.pow(1.5/utility_gamma*(strategy*strategy*(1-strategy)*(1-strategy))[:,0,:].view(num_stocks,1,npaths)*cost[:,0,:].view(num_stocks,1,npaths),1/3)
    stra_NTR_the = simpleNTR(strategy, returns, upper, lower)
    # create a new model
    model3 = NTA.WealthRNN2(input_size, hidden_size, n_layers, batch_size, seq_length,dim_size).to(device)
    # load saved data
    model3.load_state_dict(model_state_dict)
    # outputs of testing data
    _, outputs = model3(strategy.to(device),strategy[:,0,0], returns, cost, None)
    # condidence interval
    CI = 1/T*np.log(np.power(confidence_interval(np.power(outputs.detach().cpu().numpy(),1-utility_gamma),confidence),1/(1-utility_gamma)))

    return cal_esr(strategy,returns,cost,utility_gamma, T),cal_esr(stra_NTR_the,returns,cost,utility_gamma, T),cal_esr(_,returns,cost,utility_gamma, T),CI