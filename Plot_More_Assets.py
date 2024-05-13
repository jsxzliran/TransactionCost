import numpy as np
import Utility_Loss as UL
import torch
import datetime
import random
import NN_More_Assets as NMA
import Func_More_Assets as FMA
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.nn import functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_loss_esr(n_epochs, losses, utility_gamma,T,x='Epochs'):
    # Plot loss curve
    epochs = range(n_epochs+1)
    fig, ax = plt.subplots()
    ax.plot(epochs, losses)
    ax.set(xlabel=x, ylabel='Loss',
    title='loss')
    ax.grid()
    fig.savefig('loss')
    plt.show()
    
    rates = np.log(np.power((utility_gamma-1)*losses+1,1/(1-utility_gamma)))/T*100
    fig, ax = plt.subplots()
    ax.plot(epochs, rates)
    ax.set(xlabel=x, ylabel='ESR(%)',
    title='ESR')
    ax.grid()
    fig.savefig('ESR')
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
    ax.set_ylim(0.9, 1.1)
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
    fig.savefig("ESR_2.png",dpi=500,pad_inches = 0.1,bbox_inches = 'tight')
    plt.show()

def plot_ntr_two(model, Merton_opt, x_range=None, y_range=None):
    # figure out the a and b of each relu boundary
    a1=Merton_opt[0]-model.rnn.hidden_param.bias[0].cpu().detach().numpy()
    b1=Merton_opt[0]+model.rnn.hidden_param.bias[0].cpu().detach().numpy()
    a2=Merton_opt[1]-model.rnn.hidden_param.bias[1].cpu().detach().numpy()
    b2=Merton_opt[1]+model.rnn.hidden_param.bias[1].cpu().detach().numpy()
    centre = Merton_opt.reshape(2,1)
# Define the boundary matrix to plot
    x = np.array([a1,b1,a1,b1])
    y = np.array([a2,a2,b2,b2])
    c = model.rnn.rotate_param.weight.cpu().detach().numpy()
    xy = np.array([x,y])
    xy2 = np.dot(c,xy-centre)+centre
    xy3 = np.delete(xy2,1,1)
    xy4 = np.delete(xy2,2,1)

    plt.figure(figsize=(6,5))
    plt.plot(xy3[0,:], xy3[1,:],color="blue")
    plt.plot(xy4[0,:], xy4[1,:],color="blue")
    plt.grid()

    if x_range is not None:
        plt.xlim(x_range)
    
    if y_range is not None:
        plt.ylim(y_range)
        
    plt.title(label='No trade Region')
    #plt.title(label='Example: No trade Region')
    
    plt.savefig("ntr_2.png", format='png', dpi=500)
    plt.show()


def plot_ntr_two2(model, Merton_opt, x_range=None, y_range=None):
    # figure out the a and b of each relu boundary
    rotate_matrix = model.rnn.rotate_param.weight.cpu().detach().numpy()
    mat3 = np.ones([2,4])
    mat3[:,0] = np.array([-1,0])
    mat3[:,1] = np.array([0,1])
    mat3[:,2] = np.array([0,-1])
    mat3[:,3] = np.array([1,0])
    xy2 = (np.matmul(rotate_matrix,mat3).T+Merton_opt).T
    
# Define the boundary matrix to plot
    xy3 = np.delete(xy2,1,1)
    xy4 = np.delete(xy2,2,1)

    plt.figure(figsize=(6,5))
    plt.plot(xy3[0,:], xy3[1,:],color="blue")
    plt.plot(xy4[0,:], xy4[1,:],color="blue")
    plt.grid()

    if x_range is not None:
        plt.xlim(x_range)
    
    if y_range is not None:
        plt.ylim(y_range)
        
    plt.title(label='No trade Region')
    #plt.title(label='Example: No trade Region')
    
    plt.savefig("ntr_2.png", format='png', dpi=500)
    plt.show()

# a plot of two dimension using monte carlo plots
def plot_ntr_two3(model, Merton_opt, returns, cost, num_stocks,seq_length,npaths):
    #create a very random strategy
    ex_strategy = 3*torch.rand([num_stocks,seq_length,npaths]).to(device).view([num_stocks,seq_length,npaths])
    # model output an adjusted strategy
    o = model(ex_strategy, Merton_opt, returns, cost)[0]
    temp = random.randint(1,seq_length-1)
    x = o.cpu().detach().numpy()[0,temp,:].reshape([1,npaths])
    y = o.cpu().detach().numpy()[1,temp,:].reshape([1,npaths])
    plt.figure(figsize=(6,5))
    plt.grid()
    plt.title(label='No trade Region')
    plt.plot(x, y, 'o', color='blue')

    plt.savefig("ntr_2.png", format='png', dpi=500)
    plt.show()