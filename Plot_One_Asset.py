import numpy as np
import Utility_Loss as UL
import torch
import datetime
import NN_One_Asset as NOA
import Func_One_Asset as FOA
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.nn import functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Plot loss curve
def plot_loss(n_epochs, losses, x='Epochs', y='Loss',ttl = 'Loss', name = "loss_one.png" ):
    # Plot loss curve
    epochs = range(n_epochs)

    fig, ax = plt.subplots()
    ax.plot(epochs, losses)

    ax.set(xlabel=x, ylabel=y,
       title=ttl)
    ax.grid()
    fig.savefig(name)
    plt.show()

# Plot loss curve
def loss_esr(n_epochs, losses, gamma,T ,x='Epochs', y='ESR(%)',ttl = 'Equivalent Safe Rate', name = "esr_one.png" ):
    # Plot loss curve
    epochs = range(n_epochs)
    rates = np.log(np.power((gamma-1)*losses+1,1/(1-gamma)))/T*100
    fig, ax = plt.subplots()
    ax.plot(epochs, rates)

    ax.set(xlabel=x, ylabel=y,
       title=ttl)
    ax.grid()
    fig.savefig(name)
    plt.show()

def plot_esr(x, theo, lower, upper, simulated):
    # Your data
    fig, ax = plt.subplots()
    x= x*100
    theo = theo*100
    lower = lower*100
    upper = upper*100
    simulated = simulated*100
    # log
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    # plot
    asymptotic, = ax.plot(x, theo, label='Asymptotic',color='red',  
     linewidth=1, markersize=12)
    CI = ax.fill_between(x, lower, upper, alpha=0.2)
    Cmean, = ax.plot(x, simulated, 'k--',color='blue')
    # Formatter
    #ax.xaxis.set_minor_formatter(mticker.LogFormatter())
    #ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    # axis and legend
    ax.legend([asymptotic, (CI,Cmean)], ["Asymptotic", "CI+RNN"], loc=3)
    ax.set_xlabel(r'$\varepsilon$(%)')
    ax.set_ylabel(r'$ESR$(%)')
    ax.grid()
    # save
    fig.savefig("ESR_1.png",dpi=500,pad_inches = 0.1,bbox_inches = 'tight')
    plt.show()

def plot_ntr(model,lower,upper,Merton_opt,trading_cost,gamma,delta):
    # Data for plotting
    lowerbound = lower
    upperbound = upper
    optimal_input = Merton_opt
    t = torch.tensor(np.arange(lowerbound, upperbound, 0.01)).to(device)
    s = model.rnn.fc2_param.weight*F.relu(model.rnn.fc1_param.weight*F.relu(optimal_input*model.rnn.input_param.weight+model.rnn.hidden_param.weight*t-Merton_opt+model.rnn.hidden_param.bias)+2*model.rnn.hidden_param.bias)+Merton_opt+model.rnn.hidden_param.bias
    delta_real = np.power(3/(2*gamma)*np.power(Merton_opt,2)*np.power(1-Merton_opt,2)*trading_cost,1/3)+((5-2*gamma)/(10*gamma)*Merton_opt*(Merton_opt)-3/(20*gamma))*2*trading_cost
    s2 = model.rnn.fc2_param.weight*F.relu(model.rnn.fc1_param.weight*F.relu(optimal_input*model.rnn.input_param.weight+model.rnn.hidden_param.weight*t-(Merton_opt-delta_real))+2*delta_real)+Merton_opt+delta_real
    s3 = model.rnn.fc2_param.weight*F.relu(model.rnn.fc1_param.weight*F.relu(optimal_input*model.rnn.input_param.weight+model.rnn.hidden_param.weight*t-(Merton_opt-delta))+2*delta)+Merton_opt+delta

    fig, ax = plt.subplots()
    ax.plot(t.cpu().detach(), s.cpu().view(-1).detach(), label='RNN', color = 'blue', linestyle= '--', linewidth = 1)
    ax.plot(t.cpu().detach(), s2.cpu().view(-1).detach(), label='Asymptotic', color = 'red', linewidth = 1)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend()

    ax.set(xlabel=r'$\pi_{t-1}$', ylabel=r'$\pi_t$',
           title='No trade region')
    ax.grid()

    # save the plot
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%d-%m-%Y_%H-%M-%S")
    file_name = f"result__one_{formatted_time}.png"
    fig.savefig(file_name,dpi=500,pad_inches = 0.1,bbox_inches = 'tight')
    plt.show()