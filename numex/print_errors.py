import os.path
import pickle 
from math import *
import numpy as np

# dirname = os.path.dirname(__file__)

##################################################################

# problem = 1
# Ncell = 1
# incl = 1
# omega_v = [1,2]
# maxH = 0.2
# order_v = [2]
# Edge_modes = [1,2,3]

##################################################################

# folder = "omega_sweep_l2_errors"
# pickle_name = ""
# if problem == 1:
#     pickle_name += "circle_"
# else:
#     pickle_name += "crystal_"

# pickle_name += "omega_"
# for om in omega_v:
#     pickle_name += str(om) + "_"

# pickle_name += "maxH_" + str(maxH) + "_"
# pickle_name += "order_" + str(order_v[0])


def SaveTable(pickle_name, dirname, folder):
    load_file = os.path.join(dirname, folder + "/" + pickle_name + ".out")
    picklefile = open(load_file, "rb")
    omega_v, maxH, order_v, Edge_modes, relerr = pickle.load(picklefile)

    relerr_reshaped = np.reshape(relerr, (np.size(omega_v), np.size(Edge_modes)))
    # print(relerr_reshaped)

    save_file = open(os.path.join(dirname, folder + "/" + pickle_name + "_table.out"), "w")
                        
    line = "Ie" + "\t"
    for om in omega_v:
        line += "k_" + str(om) + "\t"

    line = line[:-1] + "\n"
    save_file.write(line)

    line = ""
    nr_oms = len(omega_v)
    for i, ie in enumerate(Edge_modes):
        line = str(ie) + "\t"
        for ii in range(nr_oms):
            line +=str(relerr_reshaped[ii,i]) + "\t"
        line = line[:-1] + "\n"
        save_file.write(line) 
    save_file.close()  
