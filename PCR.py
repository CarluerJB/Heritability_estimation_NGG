from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utilities import normalise_r2

def computePCR(X : pd.DataFrame, Y, n_component=25, adjusted_r2=False):
    pcr = make_pipeline(StandardScaler(), PCA(n_components=n_component), LinearRegression())
    pcr.fit(X, Y)
    if adjusted_r2:
        return normalise_r2(pcr.score(X,Y), n_component, len(X.index))
    else:
        return pcr.score(X, Y)

def computePCR_Mano(X, Y, n_component=25, adjusted_r2=False):
    X = X.to_numpy()
    XXt = np.matmul(X, X.transpose())
    eigens = np.linalg.eig(XXt)
    idx_sort = np.argsort(eigens[0])[::-1]
    eigenvalues = eigens[0][idx_sort]
    eigenvectors = eigens[1][:, idx_sort]
    newX = np.matmul(X, np.matmul(X.transpose(), eigenvectors))
    r2 = []
    for beta in np.arange(n_component):

        alpha = 0.01
        # tester : [10-2, 10-1, 1, 10, 100, 1000]


        reg = np.linalg.solve(
            np.matmul( (newX[:, :beta]).transpose(), newX[:, :beta] ) + np.eye(1) * alpha,
            np.matmul((newX[:, :beta]).transpose(), Y)
        )

        Y_tilde = np.matmul(newX[:, :beta], reg)
        if(adjusted_r2):
            r2.append(
            1 - np.sum((Y - Y_tilde)**2) / (np.sum(Y**2)) * (Y.shape[0] - 1) / (Y.shape[0] - beta - 1)
        )
        else:
            r2.append( 1 - np.sum((Y - Y_tilde)**2) / (np.sum(Y**2)) )
    return r2

def computePCR_plot(X : pd.DataFrame, Y, n_component=25, outplot_path="", title="1D", debug=False, adjusted_r2=False):
    # pcr_score=[]
    pcr_score = computePCR_Mano(X, Y, n_component=n_component, adjusted_r2=adjusted_r2)
    # for i in range(1, (n_component+1)):
    #     pcr_score.append(computePCR(X, Y, i, adjusted_r2))
        # pcr = make_pipeline(StandardScaler(), PCA(n_components=i), LinearRegression())
        # pcr.fit(X, Y)
        # if adjusted_r2:
        #     pcr_score.append(normalise_r2(pcr.score(X,Y), i, len(X.index)))
        # else:
        #     pcr_score.append(pcr.score(X, Y))
    plt.figure()
    plt.plot(list(range(1, (n_component+1))), pcr_score)
    plt.xlabel("Number of components")
    if adjusted_r2:
        plt.ylabel("adjusted R2")
    else:
        plt.ylabel("R2")
    plt.title(title)
    if debug!=True:
        plt.savefig(outplot_path)
        plt.close('all')
    else:
        plt.show()
    return pcr_score