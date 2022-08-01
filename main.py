from doctest import OutputChecker
from LINEAR_REG import *
from utilities import *
import pandas as pd
import numpy as np
import os
import sys

DEBUG = False

# PARAMETERS
parameters = sys.argv[1:]
args = {key : value for key, value in zip(parameters[0::2], parameters[1::2])}
ktop_1D = int(args['-1D']) if '-1D' in args.keys() else 7500
ktop_2D = int(args['-2D']) if '-2D' in args.keys() else 2500
ADJUSTED_R2 = args['-adjusted']=="True" if '-adjusted' in args.keys() else False
n_component_max = int(args['-nc']) if '-nc' in args.keys() else 25
method = args['-method'] if '-method' in args.keys() else "PCR"
X_matrix_path = args['-x'] if '-x' in args.keys() else None
phenotype_list = args['-phenotypes'] if '-phenotypes' in args.keys() else None
Y_vector_dir = args['-y'] if '-y' in args.keys() else None
out_path = args['-out'] if '-out' in args.keys() else None
theta_1D_path = args['-theta1D'] if '-theta1D' in args.keys() else None
theta_2D_path = args['-theta2D'] if '-theta2D' in args.keys() else None

# PARAMETERS CHECKING + OP SPE
assert n_component_max>0, "Max number of component shouldn't be lass than 1"
assert method in ["PCR", "PLS"], "Your choosen method should be PCR or PLS" 
assert X_matrix_path != None, "You must supply a genotype matrix in order to build sub-matrix"
assert Y_vector_dir != None, "You must supply a phenotype path in order to open phenotype in phenotype list"
assert phenotype_list != None, "You must supply phenotype iddentifier (at least for figure labels)"
assert (theta_1D_path != None) and ('*' in theta_1D_path), "You must supply nth-element 1D path with * for the phenotype specified in phenotype list"
assert (theta_2D_path != None) and ('*' in theta_2D_path), "You must supply nth-element 2D path with * for the phenotype specified in phenotype list"


phenotype_list = phenotype_list.split(';')

if X_matrix_path.split('.')[1]=="npy":
    X = np.load(X_matrix_path)
    X = pd.DataFrame(X)
else:
    X = pd.read_table(X_matrix_path, sep=' ', header=None)
X_center = X.sub(X.mean())
X = X_center.div(X.std())
nb_SNP = len(X.columns)
nb_indiv = len(X.index)
assert (ktop_1D<nb_indiv)==(ktop_2D<nb_indiv), "ktop1D < nb individuals and ktop2D > nb individuals : Can't compare mixed methods !"
theta_1D_path = theta_1D_path.split("*")
theta_2D_path = theta_2D_path.split("*")

for phenotype in phenotype_list:
    
    Y_vector_path = Y_vector_dir + phenotype + ".txt"
    Y = np.asarray(np.loadtxt(Y_vector_path))
    Y = normalise(Y)
    print("Working on phenotype : ", phenotype)

    # STD MODE
    Theta_ID = np.asarray(np.loadtxt(theta_1D_path[0] + phenotype + theta_1D_path[1] + ".ktop", dtype=int))
    Theta_hat_ID = pd.read_table(theta_2D_path[0] + phenotype + theta_2D_path[1] + ".ktop", header=None, names=['K'], dtype={'K' : int})
    Theta = np.asarray(np.loadtxt(theta_1D_path[0] + phenotype + theta_1D_path[1] + ".kval"))
    Theta_hat = pd.read_table(theta_2D_path[0] + phenotype + theta_2D_path[1] + ".kval", header=None)[0].to_numpy()
    X_gen, Xp, Z, Theta, Theta_hat, Theta_ID, Theta_hat_ID = generate_X_Xp_Z_Thetas(X, Theta_hat, Theta_hat_ID, Theta, Theta_ID, ktop1D=ktop_1D, ktop2D=ktop_2D, ktop_method="BF")

    r2_score_1D, r2_score_1D_2D = estimate_heritability(X_datas=[X_gen, Xp], Y_datas=Y, method=method, n_component=n_component_max, adjusted_r2=ADJUSTED_R2)

    r2_score_1D_random_2D_random_l = []
    r2_score_1D_random_l = []
    r2_score_1D_2D_random_l = []
    for i in range(0,5):
        # 1D_random, 2D_random mode
        Theta_ID = K_random_index_generator(nb_SNP=nb_SNP, nb_ktop=20000, inter=False)['K'].to_numpy()
        Theta_hat_ID = K_random_index_generator(nb_SNP=nb_SNP, nb_ktop=20000)
        Theta = np.asarray(np.loadtxt(theta_1D_path[0] + phenotype + theta_1D_path[1] + ".kval"))
        Theta_hat = pd.read_table(theta_2D_path[0] + phenotype + theta_2D_path[1] + ".kval", header=None)[0].to_numpy()
        X_gen, Xp, Z, Theta, Theta_hat, Theta_ID, Theta_hat_ID = generate_X_Xp_Z_Thetas(X, Theta_hat, Theta_hat_ID, Theta, Theta_ID, ktop1D=ktop_1D, ktop2D=ktop_2D, ktop_method="BF")
        
        r2_score_1D_random, r2_score_1D_random_2D_random = estimate_heritability(X_datas=[X_gen, Xp], Y_datas=Y, method=method, n_component=n_component_max, adjusted_r2=ADJUSTED_R2)
        r2_score_1D_random_l.append(r2_score_1D_random)
        r2_score_1D_random_2D_random_l.append(r2_score_1D_random_2D_random)

        # 1D std, 2D random mode
        Theta_ID = np.asarray(np.loadtxt(theta_1D_path[0] + phenotype + theta_1D_path[1] + ".ktop", dtype=int))
        Theta_hat_ID = K_random_index_generator(nb_SNP=nb_SNP, nb_ktop=20000)
        Theta = np.asarray(np.loadtxt(theta_1D_path[0] + phenotype + theta_1D_path[1] + ".kval"))
        Theta_hat = pd.read_table(theta_2D_path[0] + phenotype + theta_2D_path[1] + ".kval", header=None)[0].to_numpy()
        X_gen, Xp, Z, Theta, Theta_hat, Theta_ID, Theta_hat_ID = generate_X_Xp_Z_Thetas(X, Theta_hat, Theta_hat_ID, Theta, Theta_ID, ktop1D=ktop_1D, ktop2D=ktop_2D, ktop_method="BF")
        
        r2_score_1D_random, r2_score_1D_2D_random = estimate_heritability(X_datas=[X_gen, Xp], Y_datas=Y, method=method, n_component=n_component_max, adjusted_r2=ADJUSTED_R2)
        r2_score_1D_2D_random_l.append(r2_score_1D_2D_random)
        
    plot_multiple_Y_plot_many_rand(list(range(1, (n_component_max+1))),
                                        y=[r2_score_1D, r2_score_1D_2D],
                                        y_rand=[r2_score_1D_random_l, r2_score_1D_2D_random_l, r2_score_1D_random_2D_random_l],
                                        outplot_path = out_path + method + "_" + phenotype + "_HE_plot.png",
                                        legend = ["1D", "1D+2D", "1D_random", "1D+2D_random", "1D_random+2D_random"],
                                        line_type=["solid", "solid", "dotted", "dashed", "dashdot"], debug=DEBUG, adjusted_r2=ADJUSTED_R2, study=method
                                        )

    if DEBUG!=True:
        save_results([r2_score_1D, r2_score_1D_2D, r2_score_1D_random_l, r2_score_1D_2D_random_l, r2_score_1D_random_2D_random_l], out_path + method + "_" + phenotype + ".txt")