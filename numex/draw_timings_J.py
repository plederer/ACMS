import os.path
import pickle 
from math import *

Js = [i**2 for i in range(2,41)] 

maxH = 0.1
order = 4
EE=4

costs_J = [] 
costs_JlogJ = [] 
costs_JJ = [] 
times = []
JJ = []

dirname = os.path.dirname(__file__)

times_solve = []
times_assemble = []
times_extension = []
times_total = []
times_setup = []


folder = "timings_J"

for Ncell in Js:
    Nx = int(sqrt(Ncell))
    Ny = Nx   

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
        costs_J.append(J)
        costs_JJ.append(J**(1.5)/1e5)
        # costs_JlogJ.append(J*log(J)/1e1)
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
        times_assemble.append(timings["total_assemble"] - timings["assemble_basis"])

        # times_extension.append(timings["calc_harmonic_ext_remaining"])
        times_extension.append(timings["calc_harmonic_ext_assemble_and_inv"] + timings["assemble_basis"])

        # times_setup.append(timings["total_assemble"] + timings["total_calc_harmonic_ext"])

        print(60 * "#")
        picklefile.close()
        JJ.append(J)
    except:
        break

# print(f"{'J':<5}" + f"{'t_s':<25}" + f"{'t_a':<25}" + f"{'t_e':<25}"+ f"{'t_t':<25}")
print("J\tt_s\tt_a\tt_e\tt_t")
for i in range(len(JJ)):
    # print(f"{str(JJ[i]):<5}" + f"{str(times_solve[i]):<25}" + f"{str(times_assemble[i]):<25}" + f"{str(times_extension[i]):<25}"+ f"{str(times_total[i]):<25}")
    print(str(JJ[i])+ "\t" + str(times_solve[i])+ "\t" + str(times_assemble[i]) + "\t" + str(times_extension[i]) + "\t" + str(times_total[i]))

import matplotlib.pyplot as plt

plt.loglog(JJ, costs_J, "--", label = "O(J)")
# plt.loglog(JJ, costs_JlogJ, label = "O(J log(J))")
plt.loglog(JJ, costs_JJ, "--", label = "O(J^1.5)")
plt.loglog(JJ, times_total, label = "t_total")
plt.loglog(JJ, times_assemble, label = "total assemble")
plt.loglog(JJ, times_extension, label = "total extension")
plt.loglog(JJ, times_solve, label = "t_solve")
# plt.loglog(JJ, times_setup, label = "t_setup")
plt.legend()
plt.savefig('scalings_J_sp2.png', dpi=600)
plt.show()        
        