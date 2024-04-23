import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pickle as pkl
import os
import numpy as np
rings_andor_fw = ""

def plot(path, rings_andor_fw):
    print(os.path.exists('../pickle_files'))
    print(os.path.exists(path+rings_andor_fw))
    print(os.path.exists(path+rings_andor_fw+'/InsDel'))

    InsDel = torch.load(path + rings_andor_fw + '/InsDel', map_location=torch.device('cpu'),
                             pickle_module=pkl)
    nodeSub = torch.load(path + rings_andor_fw + '/nodeSub', map_location=torch.device('cpu'),
                            pickle_module=pkl)
    edgeSub = torch.load(path + rings_andor_fw + '/edgeSub', map_location=torch.device('cpu'),
                             pickle_module=pkl)
    loss_plt = torch.load(path + rings_andor_fw + '/loss_plt', map_location=torch.device('cpu'),
                            pickle_module=pkl)

    loss_valid_plt = torch.load(path + rings_andor_fw + '/loss_valid_plt', map_location=torch.device('cpu'),
                            pickle_module=pkl)


    print(np.shape(nodeSub))

    plt.figure(0)
    plt.plot(InsDel[:,0],label="node")
    plt.plot(InsDel[:,1],label="edge")
    plt.title('Node/Edge insertion/deletion costs')
    plt.legend()

    # Plotting Node Substitutions costs
    plt.figure(1)
    for k in range(nodeSub.shape[1]):
        plt.plot(nodeSub[:,k])
    plt.title('Node Substitutions costs')

    # Plotting Edge Substitutions costs
    plt.figure(2)
    for k in range(edgeSub.shape[1]):
        plt.plot(edgeSub[:,k])
    plt.title('Edge Substitutions costs')

    # Plotting the evolution of the train loss
    plt.figure(3)
    plt.plot(loss_plt)
    plt.title('Evolution of the train loss (loss_plt)')

    # Plotting the evolution of the validation loss
    plt.figure(4)
    plt.plot(loss_valid_plt)
    plt.title('Evolution of the valid loss')

    plt.show()
    plt.close()