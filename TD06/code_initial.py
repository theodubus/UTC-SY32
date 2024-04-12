#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:29:18 2020

@author: Julien Moreau

Source : https://docs.opencv.org/4.3.0/dc/dbb/tutorial_py_calibration.html

"""

#%%
# imports

import numpy as np
import cv2 as cv
import glob

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# à garder si utilisation de Spyder
%matplotlib auto

#%%
# Un ajout à faire pour Question 3

# SETUP
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
patternSize = (7,6)
objp = np.zeros((patternSize[1]*patternSize[0],3), np.float32)
objp[:,:2] = np.mgrid[0:patternSize[0],0:patternSize[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = sorted(glob.glob('./ocv-pics/left*.jpg'))
imagesok = []
listimg = []
listgray = []
listimgcorners = []
touspoints = np.array([]) # image des points des mires cumulés sur fond noir

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if touspoints.size == 0:
        touspoints = np.zeros(img.shape, dtype=np.uint8)
    
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, patternSize, None)
    #ret, corners = cv.findChessboardCornersSB(gray, patternSize, None) # exists since OpenCV 4.0.0 # with this, do not use cv.cornerSubPix
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imagesok.append(fname) # ne garder que les images utilisées
        listimg.append(img.copy())
        listgray.append(gray)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        # sur chaque image :
        cv.drawChessboardCorners(img, patternSize, corners2, ret)
        # sur l'image touspoints :
        # A COMPLETER (Question 3)
        
        #cv.imshow('img', img)
        #cv.waitKey(500)
        listimgcorners.append(img)
    else:
        print("Echec de la détection des coins pour l'image {} !".format(fname))
     
#cv.destroyAllWindows()


#%%
# Questions 1 et 2
# Affichage simple
        
# A FAIRE

#%%
# Question 3
# Affichage simple

# A FAIRE
        

#%%

# CALIBRATION
#ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv.CALIB_ZERO_TANGENT_DIST)

#%%
# Question 4
# Affichage 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.set_aspect('equal')

# A COMPLETER

# dessiner la caméra
ax.plot([0,0.5,0.5,0 , 0.5,-0.5,0 , -0.5,-0.5,0, -0.5,0.5],
        [0,-0.5,0.5,0 , 0.5,0.5,0, 0.5,-0.5,0, -0.5,-0.5],
        [0,2,2,0 , 2,2,0, 2,2,0, 2,2],
        color='r')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


#%%

# UNDISTORTION
img = cv.imread('./ocv-pics/left12.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
#mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
#dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)

#%%
# Question 5
# Affichage images corrigées

# A FAIRE


#%%
# Question 6
# Affichage des paramètres de la caméra

print("Matrice K = {}".format(mtx))
print("Centre de projection = {}".format(mtx[:2,2]))
print("Paramètres de distorsions (k1, k2, p1, p2, k3) = {}".format(dist))


#%%
# Question 7
# Dessin des vecteurs du modèle de distorsions radiales
# Afficher les flèches montrant les distorstion, c'est-à-dire partant des points rectifiés vers les points décalés (= d'origine)

step = 16 # calculer un point sur 16
# points sans distorsions (corrigés)
# A COMPLETER
# points avec distorsions (originaux avant correction)
# A COMPLETER

# A FAIRE


#%%
# Question 8
# RE-PROJECTION ERROR
mean_error = 0
imgpoints2 = []
for i in range(len(objpoints)):
    projpoints, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    imgpoints2.append(projpoints)
    error = cv.norm(imgpoints[i], projpoints, cv.NORM_L2)/len(projpoints)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

#%%
# Question 9
# Afficher les points détectés dans les images, ainsi que la reprojection des points 3D de la mire

# A FAIRE

