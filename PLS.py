
from sklearn.cross_decomposition import PLSRegression
from utilities import normalise_r2

def computePLS(X, Y, n_component=25, adjusted_r2=False):
    pls = PLSRegression(n_components=n_component)
    pls.fit(X, Y)
    if adjusted_r2:
        return normalise_r2(pls.score(X,Y), n_component, len(X.index))
    else:
        return pls.score(X, Y)
    