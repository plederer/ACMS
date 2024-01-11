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


def process_file(file_path: str):
 
    #Save all variables
    errors = np.load(file_path, allow_pickle=True)
    
    # Retrieve problem type - see if we have exact solution or not
    problem = errors["Dictionary"][()]["problem"][1]

    if problem == 1:
        dictionary = process_file_exact(errors, verbose = False)
        table_header, table_content, table_end = create_latex_table_exact(dictionary)

    else:
        dictionary = process_file_FEM(errors, verbose = False)
        table_header, table_content, table_end = create_latex_table_FEM(dictionary)

    return table_header, table_content, table_end






##################################################################
##################################################################
##################################################################
##################################################################


def process_file_FEM(errors, verbose = False):
    
    # # Retrieve h
#     index_h = file_path.find("meshH") + 5
#     index_us = file_path[index_h:].find("_")
#     h = float(file_path[index_h:index_h + index_us])
    
    # Retrieve variables
    h = errors["Dictionary"][()]["meshsize"][1]
    kwave = errors["Dictionary"][()]["wavenumber"][1]
    order = errors["Dictionary"][()]["order"][1]
    vertices = errors["Dictionary"][()]["vertices"][1]
    edges = errors["Dictionary"][()]["edges"][1]
    DoFs = list(errors["nDoFs"])
    
      
    # Retrieve L2 ACMS errors against FEM solution
    L2_ACMS_FEM_errors = errors["L2_Relative_error"]
    
   
    if verbose:
        print(f"{h=}")
        print(f"{kwave=}")
        print(f"{order=}")
        print(f"{vertices=}")
        print(f"{edges=}")
        print(f"{DoFs=}")
        print(f"{L2_ACMS_FEM_errors=}")
        
    assert len(order) == len(DoFs)
    
    return {
        "h"                     : h,
        "kwave"                 : kwave,
        "order"                 : order,
        "vertices"              : vertices,
        "edges"                 : edges,
        "DoFs"                  : DoFs,
        "L2_ACMS_FEM_errors"    : L2_ACMS_FEM_errors,
    }

 ##################################################################
##################################################################
##################################################################
##################################################################  


def process_file_exact(errors, verbose = False):

    # Retrieve variables
    h = errors["Dictionary"][()]["meshsize"][1]
    kwave = errors["Dictionary"][()]["wavenumber"][1]
    order = errors["Dictionary"][()]["order"][1]
    vertices = errors["Dictionary"][()]["vertices"][1]
    edges = errors["Dictionary"][()]["edges"][1]
    DoFs = list(errors["nDoFs"])
    
       
    # Retrieve L2 ACMS errors against exact solution
    L2_ACMS_errors = errors["L2_Relative_error"]
    
    # Retrieve L2 FEM errors against exact solution
    L2_FEM_errors = errors["FEMex_L2RelEr"][0]
    
    # Retrieve L2 errors of Nodal Interpolant
    L2_error_NodInterp = errors["L2Error_NodalInterpolant"][0]
    
    if verbose:
        print(f"{h=}")
        print(f"{kwave=}")
        print(f"{order=}")
        print(f"{vertices=}")
        print(f"{edges=}")
        print(f"{DoFs=}")
        print(f"{L2_ACMS_errors=}")
        print(f"{L2_FEM_errors=}")
        print(f"{L2_error_NodInterp=}")
        
    assert len(order) == len(DoFs)
    
    return {
        "h"                     : h,
        "kwave"                 : kwave,
        "order"                 : order,
        "vertices"              : vertices,
        "edges"                 : edges,
        "DoFs"                  : DoFs,
        "L2_ACMS_errors"        : L2_ACMS_errors,
        "L2_FEM_errors"         : L2_FEM_errors,
        "L2_error_NodInterp"    : L2_error_NodInterp
    }

 

    
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
        "\\begin{tabular}{ " + "c "*num_columns + "}\n" \
        "\\toprule\n & \\multicolumn{" + str(position_gamma) + "}{c}{$|S_\\Gamma|$}\\\\\n" \
        "h & $\\# V$ & DoFs & $p$ & " \
        + " & ".join(["\\multicolumn{1}{c}{" + str(e) + "}" for e in dictionary["edges"]]) \
        + " \\\\\n\\toprule\\\\\n"
    
    table_content = ""
    for p,dofs, L2_ACMS_FEM_errors in zip(dictionary["order"], dictionary["DoFs"], dictionary["L2_ACMS_FEM_errors"]):
        line = f"${dictionary['h']}$ & ${dictionary['vertices']}$ & ${dofs}$ & ${p}$ "
        
        for l2_err in L2_ACMS_FEM_errors:
            line += f"& ${number_LTX(l2_err[0])}$ "
        table_content += line + "\\\\\n"
    
    table_end = "\\bottomrule\n\\end{tabular}\n\\end{table}"
    # table = table_header + table_content + table_end
    return table_header, table_content, table_end


##################################################################
##################################################################
##################################################################
##################################################################

def create_latex_table_exact(dictionary):
    
    num_columns = 6 + len(dictionary["edges"])
    position_gamma = 6 + len(dictionary["edges"]) // 2
    table_header = "\\begin{table}[h!]\n\\caption{$\\kappa = " + str(dictionary['kwave']) + "$}\n" \
        "\\centering\\small\\setlength\\tabcolsep{0.55em}\n" \
        "\\begin{tabular}{ " + "c "*num_columns + "}\n" \
        "\\toprule\n & \\multicolumn{" + str(position_gamma) + "}{c}{$|S_\\Gamma|$}\\\\\n" \
        "h & $\\# V$ & DoFs & $p$ & $L^2_{FEM}$ & $L^2_{\mathcal{I}_h}$ & " \
        + " & ".join(["\\multicolumn{1}{c}{" + str(e) + "}" for e in dictionary["edges"]]) \
        + " \\\\\n\\toprule\\\\\n"
    
    table_content = ""
    for p,dofs, l2_ACMS_errors in zip(dictionary["order"], dictionary["DoFs"], dictionary["L2_ACMS_errors"]):
        line = f"${dictionary['h']}$ & ${dictionary['vertices']}$ & ${dofs}$ & ${p}$ " \
            f"& ${number_LTX(dictionary['L2_FEM_errors'])}$ & ${number_LTX(dictionary['L2_error_NodInterp'])}$ " 
        
        for l2_err in l2_ACMS_errors:
            line += f"& ${number_LTX(l2_err[0])}$ "
        table_content += line + "\\\\\n"
    
    table_end = "\\bottomrule\n\\end{tabular}\n\\end{table}"
    # table = table_header + table_content + table_end
    return table_header, table_content, table_end







##################################################################
##################################################################
##################################################################
##################################################################



def create_error_file(problem, kappa, maxH, order_v, Bubble_modes, Edge_modes, err_type):
    
    problem_dict = {
        1 : "PW",
        2 : "LIS",
        3 : "LBS"
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



def save_error_file(file_name, dictionary, mesh, l2_error, h1_error, dim, ndofs, dofs, gfu_fem, grad_fem):
    
    save_dir = Path('./Results') #Saves local folder name
    save_dir.mkdir(exist_ok=True) #Creates folder Results if it does not exists already
    file_path = save_dir.joinpath(file_name) # Full path where to save the results file (no .npy)
    # print(file_path)
    
    # 3 dimensional vector with a matrix in bubbles-edges for each order 
    # dim = len(order_v), len(Edge_modes), len(Bubble_modes)))
    l2_error_3d = np.reshape(l2_error, (dim))
    h1_error_3d = np.reshape(h1_error, (dim))
    dofs_3d = np.reshape(dofs, (dim))
    
    l2_error_rel_3d, h1_error_rel_3d = compute_l2_h1_relative_errors(mesh, gfu_fem, grad_fem, l2_error, h1_error, dim)
   
    
    # Save both 3d objects in the same file. They are assigned names: H1_FEM_error, H1_FEM_Relative_error
    np.savez(file_path, FileName = file_path, Dictionary = dictionary, nDoFs = ndofs, DoFs = dofs_3d, L2_error = l2_error_3d, L2_Relative_error = l2_error_rel_3d, H1_error = h1_error_3d, H1_Relative_error = h1_error_rel_3d)
    
    Errors = np.load(save_dir.joinpath(file_name + ".npz"), allow_pickle = True)
    print(Errors["FileName"])
    return Errors



##################################################################
##################################################################
##################################################################
##################################################################

def compute_l2_h1_relative_errors(mesh, gfu_ex, grad_uex, l2_error, h1_error, dim):

    #Computing norm of the exact solution to use for relative errors
    l2_norm_ex = Integrate ( InnerProduct(gfu_ex, gfu_ex), mesh, order = 10)
    h1_norm_ex = l2_norm_ex +  Integrate ( InnerProduct(grad_uex,grad_uex), mesh, order = 10)
    l2_norm_ex = sqrt(l2_norm_ex.real)
    h1_norm_ex = sqrt(h1_norm_ex.real)

    #Relative errors
    l2_error_rel = np.dot(l2_error, 1/l2_norm_ex)
    l2_error_rel_3d = np.reshape(l2_error_rel, (dim))
    h1_error_rel = np.dot(h1_error, 1/h1_norm_ex)
    h1_error_rel_3d = np.reshape(h1_error_rel, (dim))
    
    return l2_error_rel_3d, h1_error_rel_3d


##################################################################
##################################################################
##################################################################
##################################################################



def save_error_file_exact(file_name, dictionary, mesh, l2_error, h1_error, l2_error_fem, h1_error_fem, l2_error_FEMex, h1_error_FEMex, l2_error_NodInt, h1_error_NodInt, dim, ndofs, dofs, u_ex, Du_ex, gfu_fem, grad_fem):
    
    save_dir = Path('./Results') #Saves local folder name
    save_dir.mkdir(exist_ok=True) #Creates folder Results if it does not exists already
    file_path = save_dir.joinpath(file_name) # Full path where to save the results file (no .npy)
    # print(file_path)
    
    # 3 dimensional vector with a matrix in bubbles-edges for each order 
    # dim = len(order_v), len(Edge_modes), len(Bubble_modes)))
    l2_error_3d = np.reshape(l2_error, (dim))
    h1_error_3d = np.reshape(h1_error, (dim))
    dofs_3d = np.reshape(dofs, (dim))
    
    l2_error_rel_3d, h1_error_rel_3d = compute_l2_h1_relative_errors(mesh, u_ex, Du_ex, l2_error, h1_error, dim)
    l2_error_rel_fem, h1_error_rel_fem = compute_l2_h1_relative_errors(mesh, gfu_fem, grad_fem, l2_error_fem, h1_error_fem, dim)
    l2_error_rel_FEMex, h1_error_rel_FEMex = compute_l2_h1_relative_errors(mesh, u_ex, Du_ex, l2_error_FEMex, h1_error_FEMex, np.size(l2_error_FEMex))

    # Save both 3d objects in the same file. They are assigned names: H1_FEM_error, H1_FEM_Relative_error
    np.savez(file_path, FileName = file_path, Dictionary = dictionary, nDoFs = ndofs, DoFs = dofs_3d, L2_error = l2_error_3d, L2_Relative_error = l2_error_rel_3d, H1_error = h1_error_3d, H1_Relative_error = h1_error_rel_3d, FEMex_L2RelEr = l2_error_rel_FEMex, FEMex_H1RelEr = h1_error_rel_FEMex, FEM_L2Rel = l2_error_rel_fem, FEM_H1REl = h1_error_rel_fem, L2Error_NodalInterpolant = l2_error_NodInt, H1Error_NodalInterpolant = h1_error_NodInt)

    # Loading file to print errors (allow_picke means we can have strings)
    Errors = np.load(save_dir.joinpath(file_name + ".npz"), allow_pickle = True)
    

    return Errors

##################################################################
##################################################################
##################################################################
##################################################################
# print(Errors['Dictionary'][()]['order'])
    # print(Errors['Dictionary'][()]['bubbles'])
    # print(Errors['Dictionary'][()]['edges'])
    # print(Errors['Dictionary'][()]['vertices'])
    # print(Errors['Dictionary'][()]['problem'])
    # print(Errors['Dictionary'][()]['wavenumber'])
#     print("Degrees of Freedom")
#     print(Errors['DoFs'][0])
#     print("System size")
#     print(Errors['nDoFs'])    
    # print("L2 error")
    # print(Errors['L2_error'])
    # print("L2 relative error")
    # print(Errors['L2_Relative_error'])
    # print("H1 error")
    # print(Errors['H1_error'])
    # print("L2 nodal interpolant error")
    # print(Errors['L2Error_NodalInterpolant'])
    # print(Errors['H1Error_NodalInterpolant'])

