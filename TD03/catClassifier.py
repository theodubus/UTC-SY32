# -*- coding: utf-8 -*-
"""
SY32 : Vision et apprentissage
Printemps 2020

TD01 : Apprentissage automatique
"""

import numpy as np

class CatClassifier():
    """ Classifieur de chats 
    
    On distingue deux types de chat : type A (-1) et type B (+1).
    On classifie les chats en fonction de leur poids x à l'aide du classifier
    f_h défini telle que : f_h(x) = -1 si x <= h et +1 sinon.
    
    Parameters
    ----------
    
    Attributes
    ----------
    h_hat : float
            Valeur optimale de h.
    
    """

    h_hat = -np.Inf
    
    def __init__(self, h_hat=None):
        pass

    def predict(self, X, h=None):
        """
        Prédit le type de chat pour les poids X et le paramètre h
        
        Parameters
        ----------
        X : array-like of shape (n_sample, n_features)
            Poids des chats
        
        h : float (default=h_hat)
            Paramètre de décision 
        
        Returns
        -------
        y : array-like of shape (n_samples,)
            Type des chats
        """
        if h is None:
            h = self.h_hat
        
        y = np.where(X > h, 1, -1)

        return y.flatten()
        
    def err_emp(self, X, y, h=None):
        """
        Calcule l'erreur empirique de f_h sur l'ensemble (X,y).

        Parameters
        ----------
        X : array-like of shape (n_sample, n_features)
            Poids des chats
            
        y : array-like of shape (n_samples,)
            Type des chats
            
        h : float (default=h_hat)
            Paramètre de décision 

        Returns
        -------
        erreur : float
                Erreur empirique

        """
        if h is None:
            h = self.h_hat

        y_pred = self.predict(X, h)
        erreur = np.sum(y_pred != y) / y.shape[0]

        return erreur
        
    
    def fit(self, X, y):
        """
        Calcule la valeur optimale de h sur l'ensemble (X,y).
        L'attribut h_hat est mis à jour.'
        
        Parameters
        ----------
        X : array-like of shape (n_sample, n_features)
            Poids des chats
        
        y : array-like of shape (n_samples,)
            Type des chats
        
        Returns
        -------
        self : object
        """
        
        h_hats = map(lambda p: self.err_emp(X, y, p), X)
        self.h_hat = X[np.argmin(h_hats)]
        
        return self
