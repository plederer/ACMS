from datetime import datetime
from pathlib import Path
import numpy as np
import scipy.linalg

from ngsolve import *
from netgen.geom2d import *

import scipy.sparse as sp



##################################################################
##################################################################
##################################################################
##################################################################   


def process_file(file_path: str, ACMS_flag):
 
    #Save all variables
    errors = np.load(file_path, allow_pickle=True)
          
    h = errors["Dictionary"][()]["meshsize"]
    kwave = errors["Dictionary"][()]["kappa"]
    order = errors["Dictionary"][()]["order_v"]
    vertices = errors["Dictionary"][()]["vertices"]
    edges = errors["Dictionary"][()]["Edge_modes"]
    DoFs = list(errors["Dictionary"][()]["ndofs"])
       
    # Retrieve L2 ACMS errors against exact solution
    L2_ACMS_ex_errors = errors["L2_Relative_error"]
    H1_ACMS_ex_errors = errors["H1_Relative_error"]
    
    # Retrieve L2 ACMS FEM errors 
    L2_ACMS_FEM_errors = errors["FEM_L2Rel"]
    H1_ACMS_FEM_errors = errors["FEM_H1Rel"]
    
    # Retrieve L2 FEM errors against exact solution
    L2_FEMex_errors = errors["FEMex_L2RelEr"]
    H1_FEMex_errors = errors["FEMex_H1RelEr"]

    
    # Retrieve L2 errors of Nodal Interpolant
    L2_error_NodInterp = errors["L2Error_NodalInterpolant"]
    H1_error_NodInterp = errors["H1Error_NodalInterpolant"]
          
    assert len(order) == len(DoFs)
    
    dictionary_table = {
        "h"                     : h,
        "kwave"                 : kwave,
        "order"                 : order,
        "vertices"              : vertices,
        "edges"                 : edges,
        "DoFs"                  : DoFs,
        "L2_ACMS_ex_errors"     : L2_ACMS_ex_errors,
        "H1_ACMS_ex_errors"      : H1_ACMS_ex_errors,
        "L2_ACMS_FEM_errors"    : L2_ACMS_FEM_errors,
        "H1_ACMS_FEM_errors"    : H1_ACMS_FEM_errors,
        "L2_FEMex_errors"       : L2_FEMex_errors,
        "H1_FEMex_errors"       : H1_FEMex_errors,
        "L2_error_NodInterp"    : L2_error_NodInterp,
        "H1_error_NodInterp"    : H1_error_NodInterp
    }
    
    if ACMS_flag == 1:
        table_header, table_content_l2, table_separation, table_content_h1, table_end = create_latex_table_exact(dictionary_table)
    else:
        table_header, table_content_l2, table_separation, table_content_h1, table_end = create_latex_table_FEM(dictionary_table)

    return table_header, table_content_l2, table_separation, table_content_h1, table_end

    
 ##################################################################
##################################################################
##################################################################
##################################################################   
    
    
def number_LTX(x):
    from math import log10, floor
    if x == 0:
        string = f"0"
    else:
        e = floor(log10(abs(x)))
        i = x / 10**e
        string = f"{i:4.3f} {{\\cdot}} 10^{{{e}}}"
    return string




##################################################################
##################################################################
##################################################################
##################################################################


def create_latex_table_FEM(dictionary):
    
    num_columns = 4 + len(dictionary["edges"])
    position_gamma = 4 + len(dictionary["edges"]) // 2
    table_header = "\\begin{table}[h!]\n\\caption{$\\kappa = " + str(dictionary['kwave']) + "$}\n" \
        "\\centering\\small\\setlength\\tabcolsep{0.55em}\n" \
        "\\scalebox{0.7}{\n" \
        "\\begin{tabular}{ " + "c "*num_columns + "}\n" \
        "\\toprule\n & \\multicolumn{" + str(position_gamma) + "}{c}{$I_e$}\\\\\n" \
        "h & $\\# V$ & DoFs & $p$ & " \
        + " & ".join(["\\multicolumn{1}{c}{" + str(e) + "}" for e in dictionary["edges"]]) \
        + " \\\\\n\\toprule\\\\\n"
    
    table_content_l2 = ""
    for p,dofs, L2_ACMS_FEM_errors in zip(dictionary["order"], dictionary["DoFs"], dictionary["L2_ACMS_FEM_errors"]):
        line = f"${dictionary['h']}$ & ${dictionary['vertices']}$ & ${dofs}$ & ${p}$ "
        
        for l2_err in L2_ACMS_FEM_errors:
            line += f"& ${number_LTX(l2_err[0])}$ "
        table_content_l2 += line + "\\\\\n"
        
    table_content_h1 = ""
    for p,dofs, H1_ACMS_FEM_errors in zip(dictionary["order"], dictionary["DoFs"], dictionary["H1_ACMS_FEM_errors"]):
        line = f"${dictionary['h']}$ & ${dictionary['vertices']}$ & ${dofs}$ & ${p}$ "
        
        for h1_err in H1_ACMS_FEM_errors:
            line += f"& ${number_LTX(h1_err[0])}$ "
        table_content_h1 += line + "\\\\\n"
    
    table_separation = "\\bottomrule\n& \\multicolumn{" + str(position_gamma) + "}{c}{$I_e$}\\\\\n" \
                        "h & $\\# V$ & DoFs & $p$ & " \
                        + " & ".join(["\\multicolumn{1}{c}{" + str(e) + "}" for e in dictionary["edges"]]) \
                        + " \\\\\n\\toprule\\\\\n"  
    table_end = "\\bottomrule\n\\end{tabular}\n}\n\\end{table}"
    # table = table_header + table_content + table_end
    return table_header, table_content_l2, table_separation, table_content_h1, table_end


##################################################################
##################################################################
##################################################################
##################################################################

def create_latex_table_exact(dictionary):
    
    num_columns = 6 + len(dictionary["edges"])
    position_gamma = 6 + len(dictionary["edges"]) // 2
    table_header = "\\begin{table}[h!]\n\\caption{$\\kappa = " + str(dictionary['kwave']) + "$}\n" \
        "\\centering\\small\\setlength\\tabcolsep{0.55em}\n" \
        "\\scalebox{0.7}{\n" \
        "\\begin{tabular}{ " + "c "*num_columns + "}\n" \
        "\\toprule\n & \\multicolumn{" + str(position_gamma) + "}{c}{$I_e$}\\\\\n" \
        "h & $\\# V$ & DoFs & $p$ & $L^2_{FEM}$ & $L^2_{\mathcal{I}_h}$ & " \
        + " & ".join(["\\multicolumn{1}{c}{" + str(e) + "}" for e in dictionary["edges"]]) \
        + " \\\\\n\\toprule\\\\\n"
    
    table_content_l2 = ""
    for p,dofs, l2_FEM_errors, l2_NodInt, l2_ACMS_errors in zip(dictionary["order"], dictionary["DoFs"], dictionary["L2_FEMex_errors"], dictionary["L2_error_NodInterp"], dictionary["L2_ACMS_ex_errors"]):
        line = f"${dictionary['h']}$ & ${dictionary['vertices']}$ & ${dofs}$ & ${p}$ " \
            f"& ${number_LTX(l2_FEM_errors)}$ & ${number_LTX(l2_NodInt)}$ " 
                 
        for l2_err in l2_ACMS_errors:
            line += f"& ${number_LTX(l2_err[0])}$ "
        table_content_l2 += line + "\\\\\n"
        
    table_content_h1 = ""
    for p,dofs, h1_FEM_errors, h1_NodInt, H1_ACMS_errors in zip(dictionary["order"], dictionary["DoFs"], dictionary["H1_FEMex_errors"], dictionary["H1_error_NodInterp"], dictionary["H1_ACMS_ex_errors"]):
        line = f"${dictionary['h']}$ & ${dictionary['vertices']}$ & ${dofs}$ & ${p}$ " \
            f"& ${number_LTX(h1_FEM_errors)}$ & ${number_LTX(h1_NodInt)}$ " 
        
        for h1_err in H1_ACMS_errors:
            line += f"& ${number_LTX(h1_err[0])}$ "
        table_content_h1 += line + "\\\\\n"
        
        
    table_separation = "\\bottomrule\n & \\multicolumn{" + str(position_gamma) + "}{c}{$I_e$}\\\\\n" \
                "h & $\\# V$ & DoFs & $p$ & $H^1_{FEM}$ & $H^1_{\mathcal{I}_h}$ & " \
                + " & ".join(["\\multicolumn{1}{c}{" + str(e) + "}" for e in dictionary["edges"]]) \
                + " \\\\\n\\toprule\\\\\n"
    table_end = "\\bottomrule\n\\end{tabular}\n}\n\\end{table}"
    return table_header, table_content_l2, table_separation, table_content_h1, table_end





##################################################################
##################################################################
##################################################################
##################################################################



def create_error_file(variables_dictionary):
    
    Bubble_modes = variables_dictionary["Bubble_modes"]
    Edge_modes = variables_dictionary["Edge_modes"]
    order_v = variables_dictionary["order_v"]
    problem = variables_dictionary["problem"]
    kappa = variables_dictionary["kappa"]
    maxH = variables_dictionary["meshsize"]
    err_type = variables_dictionary["sol_ex"]  

    
    problem_dict = {
        1 : "PW",
        2 : "LIS",
        3 : "LBS",
        4 : "CrystalQuad",
        5 : "CrystalCirc",
    }

    err_type_dict = {
        0 : "FEMsol",
        1 : "EXACTsol"
    }

    date_time = datetime.now().strftime("%Y%m%d")
    file_name = f"L2-H1_errors_{err_type_dict[err_type]}_{problem_dict[problem]}_wave{kappa}_meshH{maxH}_o{order_v[-1]}_b{Bubble_modes[-1]}_e{Edge_modes[-1]}_{date_time}"
    # print(file_name)
    
    return file_name

##################################################################
##################################################################
##################################################################
##################################################################

def compute_l2_h1_relative_errors(mesh, gfu_ex, grad_uex, l2_error, h1_error, dim):
    if gfu_ex == 0 or grad_uex == 0 :
        l2_error_rel = np.dot(l2_error, 0)
        h1_error_rel = np.dot(h1_error, 0)
    else:
        #Computing norm of the exact solution to use for relative errors
        l2_norm_ex = Integrate ( InnerProduct(gfu_ex, gfu_ex), mesh, order = 10)
        h1_norm_ex = l2_norm_ex +  Integrate ( InnerProduct(grad_uex,grad_uex), mesh, order = 10)
        l2_norm_ex = sqrt(l2_norm_ex.real)
        h1_norm_ex = sqrt(h1_norm_ex.real)
        #Relative errors
        l2_error_rel = np.dot(l2_error, 1/l2_norm_ex)
        h1_error_rel = np.dot(h1_error, 1/h1_norm_ex)
    
    l2_error_rel_3d = np.reshape(l2_error_rel, (dim))
    h1_error_rel_3d = np.reshape(h1_error_rel, (dim))
    
    return l2_error_rel_3d, h1_error_rel_3d


##################################################################
##################################################################
##################################################################
##################################################################



def save_error_file(file_name, mesh, variables_dictionary, solution_dictionary, errors_dictionary):
    
    gfu_fem = solution_dictionary["gfu_fem"]
    grad_fem = solution_dictionary["grad_fem"]
   
    dim = variables_dictionary["dim"]
    ndofs = variables_dictionary["ndofs"]
    dofs = variables_dictionary["dofs"]
    u_ex = variables_dictionary["u_ex"]
    Du_ex = variables_dictionary["Du_ex"]
    
    l2_error_ex = errors_dictionary["l2_error_ex"]
    h1_error_ex = errors_dictionary["h1_error_ex"] 
    l2_error_fem = errors_dictionary["l2_error"] 
    h1_error_fem = errors_dictionary["h1_error"]
    l2_error_FEMex = errors_dictionary["l2_error_FEMex"]
    h1_error_FEMex = errors_dictionary["h1_error_FEMex"] 
    l2_error_NodInt = errors_dictionary["l2_error_NodInt"]
    h1_error_NodInt = errors_dictionary["h1_error_NodInt"]
    
    
    save_dir = Path('./Results') #Saves local folder name
    save_dir.mkdir(exist_ok=True) #Creates folder Results if it does not exists already
    file_path = save_dir.joinpath(file_name) # Full path where to save the results file (no .npy)
    # print(file_path)
    
    # 3 dimensional vector with a matrix in bubbles-edges for each order 
    # dim = len(order_v), len(Edge_modes), len(Bubble_modes)))
    l2_error_ex_3d = np.reshape(l2_error_ex, (dim))
    h1_error_ex_3d = np.reshape(h1_error_ex, (dim))
    l2_error_fem_3d = np.reshape(l2_error_fem, (dim))
    h1_error_fem_3d = np.reshape(h1_error_fem, (dim))
    dofs_3d = np.reshape(dofs, (dim))
    
    l2_error_rel_3d, h1_error_rel_3d = compute_l2_h1_relative_errors(mesh, u_ex, Du_ex, l2_error_ex, h1_error_ex, dim)
    l2_error_rel_fem, h1_error_rel_fem = compute_l2_h1_relative_errors(mesh, gfu_fem, grad_fem, l2_error_fem, h1_error_fem, dim)
    l2_error_rel_FEMex, h1_error_rel_FEMex = compute_l2_h1_relative_errors(mesh, u_ex, Du_ex, l2_error_FEMex, h1_error_FEMex, np.size(l2_error_FEMex))
    
    
    # The function variables give issues in the saving of the dictionary 
    variables_dictionary.pop("f")
    variables_dictionary.pop("g")
    
    # Save both 3d objects in the same file. They are assigned names: H1_FEM_error, H1_FEM_Relative_error
    np.savez(file_path, FileName = file_path, Dictionary = variables_dictionary, nDoFs = ndofs, DoFs = dofs_3d, L2_error_ex = l2_error_ex_3d, L2_error_fem =l2_error_fem_3d , L2_Relative_error = l2_error_rel_3d, H1_error_ex = h1_error_ex_3d, H1_error_fem = h1_error_fem_3d, H1_Relative_error = h1_error_rel_3d, FEMex_L2RelEr = l2_error_rel_FEMex, FEMex_H1RelEr = h1_error_rel_FEMex, FEM_L2Rel = l2_error_rel_fem, FEM_H1Rel = h1_error_rel_fem, L2Error_NodalInterpolant = l2_error_NodInt, H1Error_NodalInterpolant = h1_error_NodInt)

    # Loading file to print errors (allow_picke means we can have strings)
    Errors = np.load(save_dir.joinpath(file_name + ".npz"), allow_pickle = True)
    

    return Errors







# ##################################################################
# ##################################################################
# ##################################################################
# ##################################################################  




    # dictionary = {
    #     1            : ["The keys are: meshsize, order, bubbles, edges, vertices, problem, wavenumber."],
    #     'meshsize'   : ["The mesh size is", maxH],
    #     'order'      : ["The order of approximation is",  order_v],
    #     'bubbles'    : ["The number of bubble functions is", Bubble_modes],
    #     'edges'      : ["The number of edge modes is", Edge_modes],
    #     'vertices'   : ["The number of vertices is", mesh.nv],
    #     'problem'    : ["Chosen problem", problem],
    #     "wavenumber" : ["Chosen wavenumber is", kappa]
    # }


# def process_file_exact(errors):

#     # Retrieve variables
#     h = errors["Dictionary"][()]["meshsize"][1]
#     kwave = errors["Dictionary"][()]["wavenumber"][1]
#     order = errors["Dictionary"][()]["order"][1]
#     vertices = errors["Dictionary"][()]["vertices"][1]
#     edges = errors["Dictionary"][()]["edges"][1]
#     DoFs = list(errors["nDoFs"])
    
       
#     # Retrieve L2 ACMS errors against exact solution
#     L2_ACMS_ex_errors = errors["L2_Relative_error"]
#     H1_ACMS_ex_error = errors["H1_Relative_error"]
    
#     # Retrieve L2 ACMS FEM errors 
#     L2_ACMS_FEM_errors = errors["FEM_L2Rel"]
#     H1_ACMS_FEM_error = errors["FEM_H1Rel"]
    
#     # Retrieve L2 FEM errors against exact solution
#     L2_FEMex_errors = errors["FEMex_L2RelEr"]
#     H1_FEMex_error = errors["FEMex_H1RelEr"]

    
#     # Retrieve L2 errors of Nodal Interpolant
#     L2_error_NodInterp = errors["L2Error_NodalInterpolant"]
#     H1_error_NodInterp = errors["H1Error_NodalInterpolant"]
          
#     assert len(order) == len(DoFs)
    
#     return {
#         "h"                     : h,
#         "kwave"                 : kwave,
#         "order"                 : order,
#         "vertices"              : vertices,
#         "edges"                 : edges,
#         "DoFs"                  : DoFs,
#         "L2_ACMS_ex_errors"     : L2_ACMS_ex_errors,
#         "H1_ACMS_ex_error"      : H1_ACMS_ex_error,
#         "L2_ACMS_FEM_errors"    : L2_ACMS_FEM_errors,
#         "H1_ACMS_FEM_error"     : H1_ACMS_FEM_error,
#         "L2_FEMex_errors"       : L2_FEMex_errors,
#         "H1_FEMex_error"        : H1_FEMex_error,
#         "L2_error_NodInterp"    : L2_error_NodInterp,
#         "H1_error_NodInterp"    : H1_error_NodInterp
#     }

 





##################################################################
##################################################################
##################################################################
##################################################################


# def convergence_plots(plot_error, dofs, h1_error, mesh, Edge_modes, Bubble_modes, order_v):

#     ## Convergence plots

#     if plot_error ==1:

#         h1_error = np.reshape(h1_error, (len(order_v)*len(Edge_modes), len(Bubble_modes)))
#         dofs = np.reshape(dofs, (len(order_v)*len(Edge_modes), len(Bubble_modes)))


#         #Bubbles
#         plt.rcParams.update({'font.size':15})
#         for p in range(len(order_v)):
#             for i in range(len(Edge_modes)):
#                 plt.loglog(Bubble_modes, h1_error[p*len(Edge_modes) + i,:], label=('Edge modes=%i' %Edge_modes[i]))
#         plt.title('$H^1$ errors: increased bubbles deg=%i' %p)
#         plt.legend()
#         plt.xlabel('Bubbles')

#         #Edges
#         plt.rcParams.update({'font.size':15})
#         for p in range(len(order_v)):
#             for i in range(len(Bubble_modes)):
#                 plt.loglog(Edge_modes, h1_error[p*len(Edge_modes):(p+1)*len(Edge_modes),i], label=('Bubbles=%i' %Bubble_modes[i]))
#         plt.title('$H^1$ errors: increased edge modes deg=%i' %p)
#         plt.legend()
#         plt.xlabel('Edge modes')

#         plt.show()