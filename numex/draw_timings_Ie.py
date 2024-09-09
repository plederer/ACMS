import os.path
import pickle 
from math import *

maxH = 0.01
Ncell = 4
order = 6


Nx = int(sqrt(Ncell))
Ny = Nx   

Edge_modes = [4,8,16,32,64,128, 256, 512]

costs_IE = [] 
costs_IEIE = [] 
costs_IEIEIE = [] 
costs_IElogIE = [] 

times_solve = []
times_assemble = []
times_setup = []
times_extension = []
times_total = []

dirname = os.path.dirname(__file__)

folder = "timings_Ie"

EEs = []
for EE in Edge_modes:
    pickle_name =   "maxH:" +     str(maxH) + "_" + \
                    "Ncell:" +    str(Ncell) + "_" + \
                    "order:" +   str(order) + "_" + \
                    "Ie:" +      str(EE) + "_" + \
                    ".dat"
    try:
        load_file = os.path.join(dirname, folder + "/" + pickle_name)
        picklefile = open(load_file, "rb")
        data = pickle.load(picklefile)

        ex_data = data[0]
        timings = data[1]

        N = Nx
        nrE = 2 * N * (N+1)
        ne = ex_data["ne"]
        # print(ex_data["acmsndof"])
        print(nrE)
        J = Nx**2
        # costs.append((J*ne**4 + (J*EE)**3))
        # costs.append((nrE * EE**2*ne + J*ne**4)/1)
        # costs_IElogIE.append(EE*log(EE))
        
        costs_IE.append(EE/2)
        costs_IEIE.append(EE*EE/4e3)
        costs_IEIEIE.append(EE**2/1e7)
            # costs.append(nr)
        
        total = 0

        print(60 * "#")
        print("Ie = ", EE)
        for key in timings.keys():
            if "total" in key:
                print(f"{'time for ' + key + ': ':<35}" +  str(timings[key]))
                total += timings[key]
            

        print(f"{'total time':<35}" +  str(total))
        times_total.append(total)
        times_solve.append(timings["total_solve"])
        # times_assemble.append(timings["total_assemble"])
        # times_extension.append(timings["total_calc_harmonic_ext"])
        times_assemble.append(timings["total_assemble"] - timings["assemble_basis"])
        # times_extension.append(timings["total_calc_harmonic_ext"] + timings["assemble_basis"])
        times_extension.append(timings["calc_harmonic_ext_assemble_and_inv"] + timings["assemble_basis"])
        # times_setup.append(timings["total_assemble"] + timings["total_calc_harmonic_ext"])

        print(60 * "#")
        picklefile.close()
        EEs.append(EE)
    except:
        break


print("I\tt_s\tt_a\tt_e\tt_t")
for i in range(len(EEs)):
    # print(f"{str(JJ[i]):<5}" + f"{str(times_solve[i]):<25}" + f"{str(times_assemble[i]):<25}" + f"{str(times_extension[i]):<25}"+ f"{str(times_total[i]):<25}")
    print(str(EEs[i])+ "\t" + str(times_solve[i])+ "\t" + str(times_assemble[i]) + "\t" + str(times_extension[i]) + "\t" + str(times_total[i]))



import matplotlib.pyplot as plt

plt.loglog(EEs, costs_IE, "--", label = "O(I_e)")
plt.loglog(EEs, costs_IEIE, "--", label = "O(I_e^2)")
# plt.loglog(EEs, costs_IEIEIE, "--", label = "O(I_e^3)")
# plt.loglog(EEs, costs_IElogIE, "--", label = "O(I_e log(I_e))")
plt.loglog(EEs, times_total, label = "t_total")
plt.loglog(EEs, times_assemble, label = "total assemble")
plt.loglog(EEs, times_extension, label = "total extension")
plt.loglog(EEs, times_solve, label = "t_solve")
# plt.loglog(EEs, times_setup, label = "t_setup")
plt.legend()
plt.savefig('scalings_Ie.png', dpi=600)
plt.show()        
        