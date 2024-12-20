import numpy as np
from matplotlib import pyplot as plt

def line_plot(distances, name): 
    mean = np.mean(distances,axis=0)
    plt.plot(mean)
    plt.savefig(name)
    plt.show()