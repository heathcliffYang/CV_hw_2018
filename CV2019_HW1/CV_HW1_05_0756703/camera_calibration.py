#%% 
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)
    
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)


#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
#%%
print('Camera calibration...')
img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
#%%
"""
Write your code here
"""
# h = (H11, H12, H13, H21, H22, H23, H31, H32, H33)^T
#ax = (-x1, -y1, -1, 0, 0, 0, x2'*x1, x2'*y1, x2')^T
#ay = (0, 0, 0, -x1, -y1, -1, y2'*x1, y2'*y1, y2')^T
def calculate_H_and_V(images, corners):
    H = []
    V = []
    for i in range(images):        
        ## Calculate H for each image
        A = []
        for j in range(corners):
            x1 = objpoints[i][j][0]
            y1 = objpoints[i][j][1]
            z1 = 1
            x2 = imgpoints[i][j][0][0]
            y2 = imgpoints[i][j][0][1]
            z2 = 1
            # z2 = 1 -> x2' = x2 and y2' = y2
            ax = [-x1, -y1, -z1, 0, 0, 0, x2*x1, x2*y1, x2]
            ay = [0, 0, 0, -x1, -y1, -z1, y2*x1, y2*y1, y2]
            
            A.append(ax)
            A.append(ay)

        A = np.array(A)      
        u, eigen_value, vt = np.linalg.svd(A)       
        eigen_vector = vt.T
        
        h = eigen_vector[:,8].reshape(3, 3)
        h = h/h[2,2]      
        H.append(h)

        ## Use h elements to form each image's two equations of V
        #v1 = [h12*h11, h11*h22+h21*h12, h12*h31+h32*h11, h22*h21, h22*h31+h21*h32, h32*h31]
        #v2 = [h11^2-h12^2, 2*(h11*h21)-2*(h22*h12), 2*(h11*h31)-2*(h12*h32), h21^2-h22^2, 2*(h21*h31)-2*(h22*h32), h31^2-h32^2]       
        v1 = [h[0,1]*h[0,0], h[0,0]*h[1,1]+h[1,0]*h[0,1], h[0,1]*h[2,0]+h[2,1]*h[0,0], h[1,1]*h[1,0], h[1,1]*h[2,0]+h[1,0]*h[2,1], h[2,1]*h[2,0]]
        v2 = [h[0,0]**2-h[0,1]**2, 2*(h[0,0]*h[1,0])-2*(h[1,1]*h[0,1]), 2*(h[0,0]*h[2,0])-2*(h[0,1]*h[2,1]), h[1,0]**2-h[1,1]**2, 2*(h[1,0]*h[2,0])-2*(h[1,1]*h[2,1]), h[2,0]**2-h[2,1]**2]
        
        # stack v equations
        V.append(v1)
        V.append(v2)
        
    H = np.array(H)
    V = np.array(V)  
    return H , V

#%%
def calculate_K(V):
    VTV = np.dot(V.T,V)
    eigen_value, eigen_vector = np.linalg.eig(VTV)

    b = eigen_vector[:,5]
    # Put b elements in B matrix
    B = np.array([(b[0], b[1], b[2]), (b[1], b[3], b[4]), (b[2], b[4], b[5])])

    # Decompose
    K_inverse_T = np.linalg.cholesky(B)

    K = np.linalg.inv(K_inverse_T.T)
    K = K/K[2,2]
#    print("K",K)   
    return K

#%%
#            # of images x [ rotation axes , translation ]
# E's shape = 10 x 3 x 4 = [r1, r2, r3, t]
def calculate_E(images, K_inverse):
    E = []
    for i in range(images):
        e = np.zeros((3,4))

        r = np.dot(K_inverse, H[i])

        _lambda = 1/np.linalg.norm(r[:,0])
        r = np.dot(_lambda, r)

        #(a2b3-a3b2,a3b1-a1b3,a1b2-a2b1)
        # r1 x r2 = r3     ( cross product )
        r3 = np.array([r[1,0]*r[2,1]-r[2,0]*r[1,1], r[2,0]*r[0,1]-r[0,0]*r[2,1], r[0,0]*r[1,1]-r[1,0]*r[0,1]])
#        print("r3",r3)
        e[:,0] = r[:,0]
        e[:,1] = r[:,1]
        e[:,2] = r3
        e[:,3] = r[:,2]

#        print(e)
        E.append(e)

    E = np.array(E)
#    print(E.shape)
    
    return E


#%%
images = 10
corners = 49

H , V = calculate_H_and_V(images, corners)

K = calculate_K(V)

E = calculate_E(images, np.linalg.inv(K))


#%%

our_Vr = []
our_Tr = []
for i in range(images):
    result , _ = cv2.Rodrigues(E[i,:,0:3])
    result = result.reshape(-1,3).tolist()
    
    our_Vr.append(result)
    our_Tr.append(E[i,:,3].reshape(-1,3).tolist())
    
#print("our_Vr",our_Vr)
#print("our_Tr",our_Tr)
our_extrinsics = np.concatenate((our_Vr, our_Tr), axis=1).reshape(-1,6)
#print("our extrinsics",our_extrinsics)


"""
END
"""
#%%
# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = K
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, our_extrinsics
                                                , board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""



