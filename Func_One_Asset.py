import numpy as np
import torch
import torch.nn as nn
from torch import optim
import Data_generator as dg
import Utility_Loss as UL
import NN_One_Asset as NOA
import Plot_One_Asset as POA
from scipy.stats import norm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function to calculate the final return with trading cost with respect to a particular strategy
def cal_return(strat, returns, cost, n_iterations=5):  
    r =  strat[:-1, :]*returns
    r0 = strat[:-1, :]*returns
    for _ in range(n_iterations):
        r = r0-cost*abs((r+1)*strat[1:, :]-(returns+1)*strat[:-1, :])
    return r

class Cal_return(nn.Module):
    def __init__(self):
        
        super(Cal_return, self).__init__()

    def forward(self, strat,returns,cost):

        return cal_return(strat, returns, cost, n_iterations=5)

# Module for importance sampling
def importance_sampling(is_importance,seed,mu,sigma,s0,npaths,seq_length,gamma,T):
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
        stock = dg.OneStock(seed,mu,sigma,s0,npaths,seq_length-1,T)
        returns = torch.tensor(stock.Returns(),dtype=torch.float).to(device).transpose(0,1)
        scaler = 1
        
    else:
        # define the h
        h = (1-gamma)/gamma*mu/sigma
        # define the new drift under Q measure
        mu_importance = (1-gamma)/gamma*mu+mu
        # Simulate stock under Q measure
        stock = dg.OneStock(seed,mu_importance,sigma,s0,npaths,seq_length-1,T)
        # Returns under Q measure
        returns = torch.tensor(stock.Returns(),dtype=torch.float).to(device).transpose(0,1)
        # Extract Brownian Motion under Q measure
        BM_last = stock.BM()[-1,:]
        # The scaler
        scaler  = torch.exp(torch.tensor(-1/2*h*h*T-h*BM_last)).to(device)
    return returns, scaler
    
# Make a portfolio    
def make_portfolio(seed,mu,sigma,s0,npaths,seq_length,T,trading_cost,gamma):
    # Create a stock simulation with prices, returns
    # seed, mu, sigma, S0, paths, steps, T
    # Define the Merton_optimal strategy
    Merton_opt = mu/(sigma*sigma*gamma)
    # Define the distance for initial non trade region
    delta = np.power(np.power(Merton_opt*(1-Merton_opt)*trading_cost,2),1/3)
    #init_a = torch.tensor(Merton_opt-delta,dtype = torch.float)
    #init_b = torch.tensor(Merton_opt+delta,dtype = torch.float)
    stock = dg.OneStock(seed,mu,sigma,s0,npaths,seq_length-1,T)
    returns = torch.tensor(stock.Returns(),dtype=torch.float).to(device).transpose(0,1)
    # Create a default strategy as initial input, better use the optimal strategy without cost
    strategy = Merton_opt*torch.ones((seq_length,1),dtype=torch.float).to(device)
    # Create a trading cost
    cost =  torch.tensor(trading_cost*np.ones([seq_length-1,1]),dtype=torch.double).to(device)
    return returns, strategy, cost, Merton_opt, delta

# Make the model
def make_model(input_size, hidden_size, n_layers, npaths, seq_length, delta, gamma, learning_rate):

    model = NOA.WealthRNN(input_size, hidden_size, n_layers, npaths, seq_length).to(device)
    model.update_bias(delta)
    model.to(device)
    criterion = UL.PowerUtilityLoss(gamma)
    #model.rnn.fc2_param.bias.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    return model, criterion, optimizer


# Train and plot model
def train_model(strategy, target, returns, cost, scaler, model, criterion, optimizer,n_epochs):
    losses = np.zeros(n_epochs) 
    for epoch in range(n_epochs):
        #inputs = strategy.view(seq_length,1,1).to(device)
        fina_strat, outputs = model(strategy.double(), target, returns, cost, None)

        loss = criterion(outputs,scaler)
        #loss = criterion(outputs)/(1-gamma)+1/(1-gamma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[epoch] += loss    
    POA.plot_loss(n_epochs, losses)
    for name, param in model.named_parameters():
        print (name, param.data)
    return model, losses

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval for a given dataset assuming a normal distribution.

    Parameters:
    - data (array-like): The vector of data to calculate the confidence interval for.
    - confidence (float): The confidence level for the interval (default is 0.95).

    Returns:
    - A tuple containing the lower and upper bounds of the confidence interval.
    """
    # Convert data to a numpy array for convenience
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Using Bessel's correction
    n = len(data)
    
    # Calculate the z-score from the confidence level
    z = norm.ppf((1 + confidence) / 2)
    
    # Calculate the margin of error
    margin_of_error = z * (std / np.sqrt(n))
    
    # Calculate the confidence interval
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    return (ci_lower, ci_upper)


# Deal with ESR:
def ESR(mu,sigma,gamma,data, T,trading_cost, confidence=0.95):
    Merton_opt = mu/(sigma*sigma*gamma)
    ESR_simulated = 1/T*np.log(np.power(np.mean(np.power(data,1-gamma)),1/(1-gamma)))
    CI = 1/T*np.log(np.power(calculate_confidence_interval(np.power(data,1-gamma),confidence),1/(1-gamma)))
    ESR_opt = mu*mu/sigma/sigma/2/gamma
    ESR_real = mu*mu/sigma/sigma/2/gamma-gamma*sigma*sigma/2*np.power(trading_cost,2/3)*np.power(3/2/gamma*Merton_opt*Merton_opt*(1-Merton_opt)*(1-Merton_opt),2/3)
    return ESR_simulated, CI, ESR_opt, ESR_real
    