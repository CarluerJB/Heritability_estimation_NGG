from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utilities import normalise_r2

def computeLmMano(X, Y, adjusted_r2=False, diag_value=1):
    theta_hat = np.linalg.solve(X.transpose().dot(X) + np.diag(np.ones(len(X.columns)))*diag_value, X.transpose().dot(Y))
    y_hat = X.dot(theta_hat)
    r2 = 1-(np.sum(np.power((Y-y_hat), 2))/(np.sum(np.power(Y, 2))))
    if adjusted_r2:
        r2 = normalise_r2(r2, len(X.columns), len(X.index))
    return r2

def computeLm(X : pd.DataFrame, Y, adjusted_r2=False):
    lm = LinearRegression().fit(X, Y)
    if adjusted_r2:
        return normalise_r2(lm.score(X,Y), len(X.columns), len(X.index))
    else:
        return lm.score(X, Y)

def computeLm_plot(X : pd.DataFrame, Y, n_component=25, outplot_path="", title="1D", debug=False, adjusted_r2=False):
    lm = LinearRegression().fit(X, Y)
    if adjusted_r2:
        lm_score = [normalise_r2(lm.score(X,Y), len(X.columns), len(X.index))]*n_component
        #lm_score=[1 - (1-lm.score(X, Y))*((len(X.index)-1)/(len(X.index) - len(X.columns) - 1) )]*n_component
    else:
        lm_score=[lm.score(X, Y)]*n_component
    plt.figure()
    plt.plot(list(range(1, (n_component+1))), lm_score)
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
    return lm_score

def computeManyLmPlot(X:list, Y, outplot_path="", title="1D", debug=False, adjusted_r2=False):
    lm_score = []
    for X_tt in X:
        lm = LinearRegression().fit(X_tt, Y)
        if adjusted_r2:
            lm_score.append(normalise_r2(lm.score(X_tt,Y), len(X_tt.columns), len(X_tt.index)))
            #lm_score=[1 - (1-lm.score(X, Y))*((len(X.index)-1)/(len(X.index) - len(X.columns) - 1) )]*n_component
        else:
            lm_score.append(lm.score(X_tt, Y))
    plt.figure()
    plt.plot(list(range(0, len(lm_score)*5, 5)), lm_score)
    if adjusted_r2:
        plt.ylabel("adjusted R2")
    else:
        plt.ylabel("R2")
    plt.xlabel("Number of interraction added")
    plt.title(title)
    plt.tight_layout()
    if debug!=True:
        plt.savefig(outplot_path)
        plt.close('all')
    else:
        plt.show()
    return lm_score

def computeManyLmPlotMano(X:list, Y, outplot_path="", title="1D", debug=False, adjusted_r2=False, step=1):
    lm_score = []
    for X_tt in X:
        r2 = computeLmMano(X_tt, Y, adjusted_r2=adjusted_r2)
        lm_score.append(r2)
    plt.figure()
    plt.plot(list(range(0, len(lm_score)*step, step)), lm_score)
    if adjusted_r2:
        plt.ylabel("adjusted R2")
    else:
        plt.ylabel("R2")
    plt.xlabel("Number of interraction added")
    plt.title(title)
    plt.tight_layout()
    if debug!=True:
        plt.savefig(outplot_path)
        plt.close('all')
    else:
        plt.show()
    return lm_score