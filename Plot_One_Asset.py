import numpy as np
import Utility_Loss as UL
import NN_One_Asset as NOA
import Func_One_Asset as FOA
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

def plot_esr(x, theo, lower, upper, simulated):
    # Your data
    fig, ax = plt.subplots()
    x= x*100
    theo = theo*100
    lower = lower*100
    upper = upper*100
    simulated = simulated*100
    # log
    ax.set_xscale('log')
    ax.set_yscale('log')
    # plot
    asymptotic, = ax.plot(x, theo, label='Asymptotic',color='red',  
     linewidth=1, markersize=12)
    CI = ax.fill_between(x, lower, upper, alpha=0.2)
    Cmean, = ax.plot(x, simulated, 'k--',color='blue')
    # Formatter
    ax.xaxis.set_minor_formatter(mticker.LogFormatter())
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    # axis and legend
    ax.legend([asymptotic, (CI,Cmean)], ["Asymptotic", "CI+RNN"], loc=3)
    ax.set_xlabel(r'$\varepsilon$(%)')
    ax.set_ylabel(r'$ESR$(%)')
    ax.grid()
    # save
    fig.savefig("ESR_1.png",dpi=500,pad_inches = 0.1,bbox_inches = 'tight')
    plt.show()