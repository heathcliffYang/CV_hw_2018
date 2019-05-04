from __future__ import print_function
from argparse import ArgumentParser
import cv2
import numpy as np
import os
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import random
from scipy import optimize
# import bpy


def Eight_point_algorithm(pts1, pts2):
    """
    pts1^T F pts2
    """
    A = []
    for i in range(8):
        A.append([pts1[i,0,0]*pts2[i,0,0], pts1[i,0,0]*pts2[i,0,1], pts1[i,0,0], pts1[i,0,1]*pts2[i,0,0], pts1[i,0,1]*pts2[i,0,1], pts1[i,0,1], pts2[i,0,0], pts2[i,0,1], 1])
    A = np.array(A)
    u, s, v = np.linalg.svd(A)
    f = v.T
    F = f[:,-1].reshape(3,3)

    # det(F) = 0
    u, s, v = np.linalg.svd(F)
    # print("F shape",F)
    s[-1] = 0
    F = u @ np.diag(s) @ v
    return F


def RANSAC(src_pts, dst_pts, threshold):
    max_score = 0
    p = 0.95
    w = 0.5
    k = 0
    s = 8
    sample_times = int(np.log(1-p) / np.log(1-pow(w,s)))
    print("RANSAC will sample {} times.".format(sample_times))
    while True:
        # Randomly pick up 8 matches to compute F
        sample_id = random.sample([x for x in range(len(src_pts))], s)
        pts1 = []
        pts2 = []
        for i in sample_id:
            pts1.append(src_pts[i])
            pts2.append(dst_pts[i])

        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        F = Eight_point_algorithm(pts1, pts2)

        # Voting
        score = 0
        inliers = []
        for i in range(len(src_pts)):
            # x^T F x' = 0 ?
            num = src_pts[i] @ F @ dst_pts[i].T
            if math.fabs(num)  <= threshold:
                score += 1
                inliers.append(i)
        if score > max_score:
            max_score = score
            best_inliers = inliers
            best_F = F
            confidence = max_score / len(src_pts)
            print(k, "confidence", confidence)
            k = int(k + (sample_times - k) * confidence * 0.05)

        k += 1
        if k > sample_times:
            break

    return best_F, best_inliers


def normalized_image(img_shape, pts):
    normalized_m = np.array([[float(2/img_shape[0]), 0, -1],\
                             [0, float(2/img_shape[1]), -1],\
                             [0, 0, 1]])

    pixel_pts = np.ones((pts.shape[0], 3))
    pixel_pts[:,:2] = pts
    
    normalized_pts = np.matmul(normalized_m, pixel_pts.T).T
    return normalized_pts, normalized_m


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,ha = img1.shape
    for r, pt1, pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        # print(r)
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1, img2

# Read image
parser = ArgumentParser()
parser.add_argument("cali", type=int)
parser.add_argument("img_dir")
parser.add_argument("thr", type=float)
args = parser.parse_args()


K1 = np.array([[5426.566895, 0.678017, 330.096680],\
                [0.000000, 5423.133301, 648.950012],\
                [0.000000, 0.000000, 1.000000]])

K2 = np.array([[5426.566895, 0.678017, 387.430023],\
                [0.000000, 5423.133301, 620.616699],\
                [0.000000, 0.000000, 1.000000]])

img_path = args.img_dir+'/'
img_file_list = os.listdir(img_path)
img_list = []
for file in img_file_list:
    img = cv2.imread(img_path + file)
    print(img.shape)
    img_list.append(img)
 
# Find correspondence
sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()

kp1, des1 = sift.detectAndCompute(img_list[0], None)
kp2, des2 = sift.detectAndCompute(img_list[1], None)
matches = bf.knnMatch(des1, des2, k=2)
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])



MIN_MATCH_COUNT = 8
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,2)
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,2)

    # estimate F
    src_pts_n, T = normalized_image(img_list[0].shape, src_pts)
    dst_pts_n, T_p = normalized_image(img_list[1].shape, dst_pts)
    src_pts_n = src_pts_n.reshape(-1, 1, 3)
    dst_pts_n = dst_pts_n.reshape(-1, 1, 3)
    normalized_F, best_inliers = RANSAC(src_pts_n, dst_pts_n, args.thr) 
    ## de-normalize
    F = T.T @ normalized_F @ T_p

    # Draw epipolar line of each match
    tmp = np.ones((3,1))
    l = []
    for i in range(len(src_pts)):
        tmp[:2,0] = dst_pts[i].T
        l.append(F @ tmp)

    l = np.array(l)
    # draw
    img1, img2 = drawlines(img_list[0],img_list[1], l, src_pts, dst_pts)
    plt.subplot(121),plt.imshow(img1[:,:,::-1])
    plt.subplot(122),plt.imshow(img2[:,:,::-1])
    plt.savefig(args.img_dir+'epipolar_line.png')

    # Essential matrix 
    E = K1.T @ F @ K2
    U, D, V = np.linalg.svd(E)
    vt = V.T
    W = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])

    # 4 possible P2
    u3 = U @ np.array([[0,0,1]]).T
    P2_set = np.zeros((4, 3, 4))
    P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    X_set = np.zeros([4, len(src_pts), 4])
    fig = plt.figure()
    for i in range(4):
        if i < 2:
            P2_set[i, :, :3] = U @ W @ vt
        else:
            P2_set[i, :, :3] = U @ W.T @ vt
        if i % 2 == 0:
            P2_set[i, :, -1] = u3[:,0]
            
        else:
            P2_set[i, :, -1] = -u3[:,0]

        # Triangulation
        tmp1 = np.ones((3,1))
        tmp2 = np.ones((3,1))
        for j in range(len(src_pts)):
            A = []
            A.append(src_pts[j][0] * P1[2, :] - P1[0, :])
            A.append(src_pts[j][1] * P1[2, :] - P1[1, :])
            A.append(dst_pts[j][0] * P2_set[i, 2, :] - P2_set[i, 0, :])
            A.append(dst_pts[j][1] * P2_set[i, 2, :] - P2_set[i, 1, :])

            A = np.array(A)

            tmp1[:2, 0] = src_pts[j].T
            tmp2[:2, 0] = dst_pts[j].T
            def fun(X):
                X_full = np.ones((4,1))
                X_full[:3, 0] = X
                return np.array([(tmp1[0] - P1[0].reshape(1,4) @ X_full)**2 + (tmp2[0] - P2_set[i,0].reshape(1,4) @ X_full)**2,\
                        (tmp1[1] - P1[1].reshape(1,4) @ X_full)**2 + (tmp2[1] - P2_set[i,1].reshape(1,4) @ X_full)**2,\
                        (tmp1[2] - P1[2].reshape(1,4) @ X_full)**2 + (tmp2[2] - P2_set[i,2].reshape(1,4) @ X_full)**2]).flatten()

            def jac(X):
                X_full = np.ones((4,1))
                X_full[:3, 0] = X
                ja = np.zeros((3,3))
                for n in range(3):
                    for m in range(3):
                        ja[n,m] = -(tmp1[n] - P1[n] @ X_full)*2*P1[n,m] - (tmp2[0] - P2_set[i,n] @ X_full)*2*P2_set[i,n,m]

                return ja

            U_a, S_a, V_a = np.linalg.svd(A)
            X_set[i, j] = V_a.T[:,-1]
            sol = optimize.root(fun, X_set[i, j, :3], jac=jac, method='lm')
            X_set[i, j, :3] = sol.x

        R1 = P1[:3,:3]
        R2 = P2_set[i, :3, :3]
        view_direction1 = R1.T[2,:]
        view_direction2 = R2.T[2,:]
        front_pts_idx1 = []
        front_pts_idx2 = []
        for j in range(len(src_pts)):
            # print(X_set[i, j, :3].shape, view_direction1.shape)
            cos1 = X_set[i, j, :3].reshape(1,3) @ view_direction1.reshape(3,1)
            cos2 = X_set[i, j, :3].reshape(1,3) @ view_direction2.reshape(3,1)
            # print(cos1.shape)
            if cos1 > 0:
                front_pts_idx1.append(j)
            if cos2 > 0:
                front_pts_idx2.append(j)

        common = list(set(front_pts_idx1).intersection(front_pts_idx2))


        label = '{} pts in front of both'.format(len(common))
        if i == 0:
            ax = fig.add_subplot(221, projection='3d')
            ax.set_title(label)
        elif i == 1:
            ax = fig.add_subplot(222, projection='3d')
            ax.set_title(label)
        elif i == 2:
            ax = fig.add_subplot(223, projection='3d')
            ax.set_title(label)
        else:
            ax = fig.add_subplot(224, projection='3d')
            ax.set_title(label)

        

        ax.scatter([X_set[i,j,0] for j in common], [X_set[i,j,1] for j in common], [X_set[i,j,2] for j in common])
        
    plt.savefig(args.img_dir+'ptcloud.png')
