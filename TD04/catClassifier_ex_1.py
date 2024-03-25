# -*- coding: utf-8 -*-
"""
SY32 : Vision et apprentissage
Printemps 2020

TD02 : Apprentissage automatique
"""

import numpy as np

class CatClassifierMultiDim():
    def __init__(self):
        self.h = None
        self.d = None
        self.z = None

    def predict(self, X, d=None, z=None, h=None):
        if any([d is None, z is None, h is None]):
            d, z, h = self.d, self.z, self.h
        if any([d is None, z is None, h is None]):
            raise ValueError("Missing parameters")

        y = X[:, d] <= h
        y = y.astype(int)
        y = np.where(y, z, -z)
        return y

    def err_emp(self, X, y, h=None, d=None, z=None):
        if any([d is None, z is None, h is None]):
            d, z, h = self.d, self.z, self.h
        if any([d is None, z is None, h is None]):
            raise ValueError("Missing parameters")

        y_pred = self.predict(X, d, z, h)
        erreur = np.mean(y_pred != y)
        return erreur
        

    def fit(self, X, y):
        # On veut tester toutes les valeurs de X[d] pour h, {-1, 1} pour z et range(X.shape[1]) pour d
        y = y.flatten()
        params = [(h, d, z) for d in range(X.shape[1]) for h in X[:, d] for z in [-1, 1]]
        solutions = list(map(lambda p: self.err_emp(X, y, p[0], p[1], p[2]), params))
        index_best_params = np.argmin(solutions)
        self.h, self.d, self.z = params[index_best_params]       
        return self
