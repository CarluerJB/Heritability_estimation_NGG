import imp
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import pandas as pd
from utilities import normalise_r2

def computePLS(X : pd.DataFrame, Y, n_component=25, adjusted_r2=False):
    pls = PLSRegression(n_components=n_component)
    pls.fit(X, Y)
    if adjusted_r2:
        return normalise_r2(pls.score(X,Y), n_component, len(X.index))
    else:
        return pls.score(X, Y)
    

def computePLS_plot(X : pd.DataFrame, Y, n_component=25, outplot_path="", title="1D", debug=False, adjusted_r2=False):
    pls_score=[]
    for i in range(1, (n_component+1)):
        pls = PLSRegression(n_components=i)
        pls.fit(X, Y)
        if adjusted_r2:
            pls_score.append(normalise_r2(pls.score(X,Y), i, len(X.index)))
        else:
            pls_score.append(pls.score(X, Y))
    plt.figure()
    plt.plot(list(range(1, (n_component+1))), pls_score)
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
    return pls_score