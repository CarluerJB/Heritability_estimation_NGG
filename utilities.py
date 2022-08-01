import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def normalise_r2(r2, p, n):
    return 1 - (1-r2)*((n-1)/(n - p - 1) )

def normalise(Y):
    return (Y - np.mean(Y))/np.std(Y)

def generate_X(SNP_list, X_matrix : pd.DataFrame, inter : bool):
    if inter :
        SNP_list = generate_SNP_list_inter(SNP_list, len(X_matrix.columns))
        # return X_matrix[SNP_list['i']] * X_matrix[SNP_list['j']] DOES NOT WORK !!
        SNP1 = X_matrix[SNP_list['i']]
        SNP2 = X_matrix[SNP_list['j']]
        SNP1 = SNP1.T.reset_index(drop=True).T
        SNP2 = SNP2.T.reset_index(drop=True).T
        return SNP1.mul(SNP2)
    return X_matrix[SNP_list]

def generate_SNP_list_inter(SNP_list, size):
    SNP_list['i'] = int(size) -2 - np.floor(np.sqrt(-8*SNP_list['K'] + 4 * int(size) * (int(size)-1) - 7) / 2 - 0.5)
    SNP_list['i'] = SNP_list['i'].astype('int')
    SNP_list['j'] = SNP_list['K'] + SNP_list['i'] + 1 - int(size) * (int(size)-1) / 2 + (int(size) - SNP_list['i']) * ((int(size) - SNP_list['i']) - 1) / 2
    SNP_list['j'] = SNP_list['j'].astype('int')
    return SNP_list

def generate_X_Xp_Z_Thetas(X, Theta_hat, Theta_hat_ID, Theta, Theta_ID, usektop=True, ktop1D = 7500, ktop2D=2500, ktop_method='BF', step=1):
    print("generating data with :\n\tktop1D = " + str(ktop1D) + "\n\tktop2D = " + str(ktop2D))
    X_gen = generate_X(Theta_ID, X, False)
    if usektop:
        if ktop_method=='BF':
            Theta = Theta[:ktop1D]
            Theta_hat = Theta_hat[:ktop2D]
            Theta_hat_ID = Theta_hat_ID.head(ktop2D)
        if ktop_method=='SM':
            Theta_full = np.concatenate([Theta, Theta_hat])
            ktop_full = np.argsort(Theta_full)[-(ktop1D+ktop2D):]
            ktop1D = len(ktop_full[ktop_full<Theta_ID.shape[0]])
            ktop2D = len(ktop_full[ktop_full>Theta_ID.shape[0]])
            Theta = Theta[:ktop1D]
            Theta_hat = Theta_hat[:ktop2D]
            Theta_hat_ID = Theta_hat_ID.head(ktop2D)
            print("KTOP RATIO : " + str(ktop1D) + "/" + str(ktop2D))
        if ktop_method=="MtM":
            Theta = Theta[:ktop1D]
            Theta_hat = Theta_hat[:ktop2D]
            Theta_hat_ID = Theta_hat_ID.head(ktop2D)
            x_range = range(step, ktop2D+step, step)
            Xp_l=[]
            X_gen.drop(X_gen.columns[ktop1D:], axis=1, inplace=True)
            Xp_l.append(X_gen)
            for i in range(0, len(x_range)):
                Z = generate_X(Theta_hat_ID.head(x_range[i]), X, True)
                Xp = pd.concat([X_gen, Z], axis=1)
                Xp_l.append(Xp)
            return None, Xp_l, None, None, None, None, None
    Z = generate_X(Theta_hat_ID, X, True)
    if usektop:
        X_gen.drop(X_gen.columns[ktop1D:], axis=1, inplace=True)
    Xp = pd.concat([X_gen, Z], axis=1)
    return X_gen, Xp, Z, Theta, Theta_hat, Theta_ID, Theta_hat_ID

def plot_multiple_Y_plot(x, y=[], title="", outplot_path="",legend=[], line_type=[], debug=False, adjusted_r2=False, study = "PCR"):
    plt.figure()
    for i_elem in range(0, len(y)):
        plt.plot(x, y[i_elem], label=legend[i_elem], linestyle=line_type[i_elem])
    plt.xlabel("Number of components" if study=="PCR" else "Number of interraction added")
    if adjusted_r2:
        plt.ylabel("adjusted R2")
    else:
        plt.ylabel("R2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if debug!=True:
        plt.savefig(outplot_path)
        plt.close('all')
    else:
        plt.show()

def K_random_index_generator(nb_SNP=0, nb_ktop=0, inter=True):
    if inter:
        nb_interaction = int(nb_SNP*(nb_SNP-1)/2)
        K_id = random.sample(range(0, nb_interaction), nb_ktop)
    else:
        K_id = random.sample(range(0, nb_SNP), nb_ktop)
    K_pd = pd.DataFrame(K_id, columns = ['K'], dtype=int)
    return K_pd



def save_results(results_list, results_path):
    with open(results_path, 'w') as file:
        for results in results_list:
            file.write(str(results) + "\n")

from PCR import computePCR
from PLS import computePLS

def estimate_heritability(X_datas, Y_datas, method="PCR", n_component=25, adjusted_r2=True):
    r2_list = []
    if method=="PCR":
        for i in range(0, len(X_datas)):
            r2_list.append(computePCR(X_datas[i], Y_datas, n_component=n_component, adjusted_r2=adjusted_r2))
    elif method=="PLS":
        for i in range(0, len(X_datas)):
            r2_list.append(computePLS(X_datas[i], Y_datas, n_component=n_component, adjusted_r2=adjusted_r2))
    else:
        print("ERROR : FUNC " + method + " is unknown !")
        exit(0)
    return r2_list


def plot_multiple_Y_plot_many_rand(x, y=[], y_rand=[], title="", outplot_path="",legend=[], line_type=[], debug=False, adjusted_r2=False, study = "PCR"):
    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.figure()
    for i_elem in range(0, len(y)):
        plt.plot(x, y[i_elem], label=legend[i_elem], linestyle=line_type[i_elem], color=color[i_elem])
    i_elem = len(y)
    for elem in y_rand:
        p = plt.plot(x, elem[0], label=legend[i_elem], linestyle=line_type[i_elem], color=color[i_elem])
        for y_it in elem[1:]:
            plt.plot(x, y_it, linestyle=line_type[i_elem], alpha=0.4, color=color[i_elem])
        i_elem=i_elem+1
    plt.xlabel("Number of components" if study=="PCR" else "Number of interraction added")
    if adjusted_r2:
        plt.ylabel("adjusted R2")
    else:
        plt.ylabel("R2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if debug!=True:
        plt.savefig(outplot_path)
        plt.close('all')
    else:
        plt.show()