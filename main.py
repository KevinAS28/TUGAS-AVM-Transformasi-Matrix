import cv2 
import numpy as np
import matplotlib.pyplot as plt
import inspect

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

def get_ortho_proj_y():
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

def get_expansion_yk(k):
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
    

def get_matrices_affine(matrix_operator_list):
    identity = np.identity(2) # Membuat matrix identitas 2 dimensi

    matrice = identity

    # Mengkalikan semua matrix transformasi
    for matx in matrix_operator_list:
        matrice = matrice@matx

    # Menambah 1 kolom 0 (Harus 3 kolom karena bentuk gambar adalah R, G, B)
    extra = np.array([[0], [0]]) 
    matrice = np.append(matrice, extra, axis=1)

    return matrice

def show_image(image):
    _, ax = plt.subplots()
    ax.imshow(image)
    plt.show()

if __name__=="__main__":
    
    image_path = input('Image path: ')

    transformation_matrices = []

    names_functions = {
                    "Refleksi_x" : get_reflection_x,
                    "Refleksi_y " : get_reflection_y,
                    "Refleksi_yx" : get_reflection_yx,
                    "Ortho_x" : get_ortho_proj_x,
                    "Ortho_y" : get_ortho_proj_y,
                    "Dilatasi" : get_dilatation,
                    "Expansi_xk" : get_expansion_xk,
                    "Expansi_yk" : get_expansion_yk,
                    "Shear_xk" : get_shear_xk,
                    "Shear_yk" : get_shear_yk,
                    "Rotasi" : get_rotation

    }

    while True:
        print('Pilih transformasi matrix: ')
        print('\n'.join([f'{i+1}. {items[0]}' for i, items in enumerate(names_functions.items())]))
        pilihan = int(input('Pilihan: '))

        transformation = names_functions[list(names_functions.keys())[pilihan-1]]
        req_arguments = inspect.getfullargspec(transformation).args
        arguments = []
        for arg in req_arguments:
            arguments.append(int(input(f'Argumen {arg}: ')))

        again = input('Tambah transformasi matrix? [y/n] ').lower()=='y'

        transformation_matrices.append(transformation(*arguments))

        if not again:
            break

    # print(image_path)
    # print(transformation_matrices)

    A = get_matrices_affine(transformation_matrices)

    image = cv2.imread(image_path) # Membaca gambar dengan matrix
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert ke bentuk RGB karena default opencv membaca secara BGR
    image = cv2.resize(image, (512, 256)) # Resize image ke tinggi 512 pixel, lebar 256 pixel

    s_max = A.max() if A.max() >= 1 else max(image.shape[:2]) # Mencari background image

    warped = cv2.warpAffine(image, A, s_max) 

    show_image(warped)
