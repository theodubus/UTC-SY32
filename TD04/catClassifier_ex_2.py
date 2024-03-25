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
        self.p = None

    def predict(self, X, d=None, z=None, h=None):
        if any([d is None, z is None, h is None]):
            d, z, h = self.d, self.z, self.h
        if any([d is None, z is None, h is None]):
            raise ValueError("Missing parameters")

        y = X[:, d] <= h
        y = y.astype(int)
        y = np.where(y, z, -z)
        return y

    def err_emp(self, X, y, h=None, d=None, z=None, p=None):
        if any([d is None, z is None, h is None, p is None]):
            d, z, h, p = self.d, self.z, self.h, self.p
        if any([d is None, z is None, h is None, p is None]):
            raise ValueError("Missing parameters")

        y_pred = self.predict(X, d, z, h)
        erreur = np.sum((y_pred != y) * p)
        return erreur
        
    def fit(self, X, y, p):
        self.p = p
        y = y.flatten()
        params = [(h, d, z) for d in range(X.shape[1]) for h in X[:, d] for z in [-1, 1]]
        solutions = list(map(lambda a: self.err_emp(X, y, a[0], a[1], a[2], p), params))
        index_best_params = np.argmin(solutions)
        self.h, self.d, self.z = params[index_best_params]       
        return self


class CatClassifierBoost():
    def __init__(self):
        self.models = []
        self.alphas = []

    def predict(self, X):
        if len(self.models) == 0 or len(self.alphas) == 0:
            raise ValueError("No models fitted")
        
        y = np.sum([alpha * model.predict(X) for alpha, model in zip(self.alphas, self.models)], axis=0)
        y = np.where(y > 0, 1, -1)
        return y

    def err_emp(self, X, y):
        y_pred = self.predict(X)
        erreur = np.sum(y_pred != y) / X.shape[0]
        erreur = np.mean(y_pred != y)
    
        return erreur

    def fit(self, X, y, K=10, verbose=False):
        p = np.ones(X.shape[0]) / X.shape[0]
        k = 0
        while k < K:
            model = CatClassifierMultiDim()
            model.fit(X, y, p)
            erreur = model.err_emp(X, y, p)
            alpha = 0.5 * np.log((1 - erreur) / erreur)
            self.models.append(model)
            self.alphas.append(alpha)
            if verbose:
                print(f"Model {k} : h = {model.h}, d = {model.d}, z = {model.z}, alpha = {alpha}, erreur = {erreur}")
            # Les poids p de chaque modèle est stockée dans self.p pour chacun d'entre eux, inutile de les conserver ici
            p = (p * np.exp(-alpha * y * model.predict(X))) / (2 * np.sqrt(erreur * (1 - erreur)))
            k += 1

