from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

def transformer(I, H, hw = (-1,-1), interp='linear'):
    h = hw[0]
    w = hw[1]
    if (w <= 0 or h <= 0):
        (h,w) = hw = I.shape[:2]
    O = np.zeros((h,w)) # image de sortie

    xx1, yy1 = np.meshgrid(np.arange(I.shape[1]), np.arange(I.shape[0]))
    xx1 = xx1.flatten()
    yy1 = yy1.flatten()

    Hinv = np.linalg.inv(H)

    xx2, yy2 = np.meshgrid(np.arange(O.shape[1]), np.arange(O.shape[0]))
    xx2 = xx2.flatten()
    yy2 = yy2.flatten()

    xxyy2 = np.stack((xx2,yy2,np.ones((O.size))), axis=0)

    xxyy = Hinv @ xxyy2
    xxyy = np.stack((xxyy[0]/xxyy[2], xxyy[1]/xxyy[2]), axis=0)

    O = griddata((xx1,yy1), I.flatten(), xxyy.T, method=interp, fill_value=0).reshape(O.shape)

    return O

def two_plots(title1, img1, title2, img2, cmap1='gray', cmap2='gray'):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img1, cmap=cmap1)
    ax[0].set_title(title1)
    ax[1].imshow(img2, cmap=cmap2)
    ax[1].set_title(title2)
    plt.show()

def select_corners(image):
    plt.imshow(image)
    coins = plt.ginput(4, timeout=0)
    plt.close()

    # On reordonne les coins ce qui permet de les séléctionner dans n'importe quel ordre dans ginput
    # Ne fonctionnera pas correctement si séléction trop cisaillée (ex: coin haut gauche (et droit) plus à droite que coin bas droit) 
    coins = sorted(coins, key=lambda x: x[1])
    coins = [list(coin) for coin in coins]
    if coins[0][0] > coins[1][0]:
        coins[0], coins[1] = coins[1], coins[0]
    if coins[2][0] < coins[3][0]:
        coins[2], coins[3] = coins[3], coins[2]
    
    return coins