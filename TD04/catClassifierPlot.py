# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:12:43 2020

@author: xuphilip
"""

import numpy as np
import matplotlib.pyplot as plt

def plotCatClassifier(clf, xlim=(0,10), ylim=(40, 56), xstep=0.1, ystep=0.1):
    # Génère une grille d'instances de X
    x = np.arange(xlim[0], xlim[1], xstep)
    y = np.arange(ylim[0], ylim[1], ystep)
    x_ = np.repeat(x, len(y))
    y_ = np.tile(y, len(x))
    X = np.column_stack((x_,y_))
    # Prédit la classe de X
    Y = clf.predict(X)
    # Trace le graphe de prédiction
    plt.figure()
    plt.plot(X[Y<0,0], X[Y<0,1], 'rs')
    plt.plot(X[Y>0,0], X[Y>0,1], 'bs')
    plt.legend(('Type A', 'Type B'))
    plt.show()