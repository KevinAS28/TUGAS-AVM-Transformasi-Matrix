
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(image):
    _, ax = plt.subplots()
    ax.imshow(image)
    plt.show()

def get_reflection_y():
    return np.array([
        [-1, 0],
        [0, 1]
        ])

def get_reflection_x():
    return np.array([
        [1, 0],
        [0, -1]
        ])

def get_reflection_yx():
    return np.array([
        [0, 1],
        [1, 0]
        ])

def get_ortho_proj_x():
    return np.array([
        [1, 0],
        [0, 0]
        ])

def get_ortho_proj_x():
    return np.array([
        [0, 0],
        [0, 1]
        ])

def get_dilatation(k):
    return np.array([
        [k, 0],
        [0, k]
        ])

def get_expansion_xk(k):
    return np.array([
        [k, 0],
        [0, 1]
        ])

def get_expansion_xk(k):
    return np.array([
        [1, 0],
        [0, k]
        ])

def get_shear_xk(k):
    return np.array([
        [1, k],
        [0, 1]
        ])

def get_shear_yk(k):
    return np.array([
        [1, 0],
        [k, 1]
        ])

def get_rotation(degree):
    r = np.radians(degree)
    sin_theta = np.sin(r)
    cos_theta = np.cos(r)
    
    return np.array([
        [cos_theta, sin_theta],
        [-sin_theta, cos_theta]
    ])
    

def get_affine_cv(matrix_operator_list):
    identity = np.identity(2)

    matrice = identity

    for matx in matrix_operator_list:
        matrice = matrice@matx

    extra = np.array([[0], [0]])
    
    matrice = np.append(matrice, extra, axis=1)
    return matrice

A2 = get_affine_cv([get_shear_xk(1), get_rotation(-45), get_reflection_x()])
print(A2)

print(A2.shape)

image_path = 'bag.png'
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (512, 256))

s_max = A2.max() if A2.max() >= 1 else max(image.shape[:2])
warped = cv2.warpAffine(image, A2, (1000, 1000))
show_image(warped)

"""
INPUT:
pilih gambar
pilih transformasi
"""

