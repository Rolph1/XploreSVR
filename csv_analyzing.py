import csv
import numpy as np
from scipy.sparse import csr_matrix

import numpy as np
import math

#optional preprocessing used for experimenting with very sparse
# vectors, not relevant with a vector that is only 1792 elements long.
'''
def preprocessing(X):
    # Convert the entire dataset to a sparse matrix first
    X_sparse = csr_matrix(X)
    
    # Initialize Truncated SVD with 300 components
    # Note: Ensure that n_components is less than min(X.shape)
    #svd = TruncatedSVD(n_components=min(5, np.shape(np.array(X))[1]-1)) 
    
    # Apply SVD on the sparse matrix
    X_transformed = svd.fit_transform(X_sparse)
        # Initialize the Min-Max scaler
    #scaler = MinMaxScaler()
    # Scale the transformed data to range [0, 1]
    #X_scaled = scaler.fit_transform(X_transformed)
    #min(300, np.shape(np.array(X))[1]-1)
    # Print shape to confirm it's 2D
    #print(np.shape(X_scaled))
    #print(np.array(X_scaled))
    return X_transformed
'''

def normalize_rgb(x):
    return int(x)/255

# I implemented this based on the following page:
# https://www.rmuti.ac.th/user/kedkarn/impfile/RGB_to_HSI.pdf
def rgb_to_hsi(R, G, B):
    """
    Converts rgb values to hsi
    H is in [0, 360]
    S is in [0, 100]
    I is in [0, 255]
    """
    r, g, b = R / 255.0, G / 255.0, B / 255.0
    i = (r+g+b) / 3
    min_value = min(r, g, b)
    s = 1 - (min_value) / (i + 1e-6) if i != 0 else 0
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g)**2 + (r - b) * (g - b))
    h = np.arccos(num / (den + 1e-6)) if den != 0 else 0
    if b > g:
        h = 2*np.pi - h
    H = h*180 / np.pi   #[0, 360]
    S = s*100           #[0,100]
    I = i*255           #[0,255]
    return H, S, I

def grayscale(r, g, b):
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def find_occurences(X):
    """
    Creates a vector of length 1792, with the first 768 elements being the occurences of
    colors in the RGB color space, the second 768 elements being the occurences in the HSI 
    color space, and finally the last 256 colors are the occurencses in the grayscale spectrum.
    """
    # Create a vector for all occurrences: RGB (768), HSI (768), and Grayscale (256)
    RGB_LENGTH = 768
    HSI_LENGTH = 768
    GRAYSCALE_LENGTH = 256

    RGB_BIN_DIVIDE = 12
    HSI_H_MAX = 360
    HSI_S_MAX = 100
    HSI_I_MAX = 255

    HSI_H_DIVIDE = 4
    HSI_S_DIVIDE = 6
    HSI_I_DIVIDE = 32

    Y = [0] * (RGB_LENGTH + HSI_LENGTH + GRAYSCALE_LENGTH)

    for r, g, b in X:
        # Putting RGB Indices in the vector
        r_index = int(r // (RGB_LENGTH / RGB_BIN_DIVIDE))
        g_index = int(g // (RGB_LENGTH / RGB_BIN_DIVIDE))
        b_index = int(b // (RGB_LENGTH / RGB_BIN_DIVIDE))
        rgb_index = r_index * (12**2) + g_index * 12 + b_index
        Y[rgb_index] += 1
        
        # HSI Indices
        H, S, I = rgb_to_hsi(r, g, b)

        # prevent overflow
        h_index = int(H / (HSI_H_MAX/HSI_H_DIVIDE))  # Divides 360 by 4
        s_index = int(S / (HSI_S_MAX/HSI_S_DIVIDE))  # Divides 100 by 6
        i_index = int(I / (HSI_I_MAX/HSI_I_DIVIDE))  # Divides 100 by 32

        if h_index > HSI_H_DIVIDE - 1:
            h_index = HSI_H_DIVIDE - 1
        if s_index > HSI_S_DIVIDE - 1:
            s_index = HSI_S_DIVIDE - 1
        if i_index > HSI_I_DIVIDE - 1:
            i_index = HSI_H_DIVIDE - 1

        hsi_index = 768 + h_index * (6 * 32) + s_index * 32 + i_index
        if hsi_index >= (768 + 768) or hsi_index < 768:
            raise ValueError(f"Calculated HSI index is out of bounds: {hsi_index}, H: {H}, S: {S}, I:{I}")

        Y[hsi_index] += 1

        # Grayscale Index
        gray_value = int(grayscale(r, g, b))
        gray_index = 1536 + math.ceil(gray_value)
        Y[gray_index] += 1
    
    return Y


def calculate_features(A):
    """
    Used when experimenting with alternative ways of pooling/
    training the model, not relevant for the final product.
    """
    r_ = [i[0] for i in A]
    g_ = [i[1] for i in A if len(i) > 1]
    b_ = [i[2] for i in A if len(i) > 2]

    r_mean = np.mean(r_, axis=0)
    g_mean = np.mean(g_, axis=0)
    b_mean = np.mean(b_, axis=0)
    r_std = np.std(r_, axis=0)
    g_std = np.std(g_, axis=0)
    b_std = np.std(b_, axis=0)
    #percentiles = np.percentile(A, [25, 50, 75], axis=0)
    return [r_mean, g_mean, b_mean, r_std, g_std, b_std]


def read_rows(filename, rows, X, fun):
    """
    Reads select rows of a given file and places them in the array
    X

    Parameters
    ----------
    filename : str
        The filename
    rows : int[]
        array containing the index of the rows to be read
    X : int[]
        array that contains the values that have been read
    fun :
        function to be applied on the element before being added to the array
    arr :
        boolean that decides if each row is placed as an array or just appended
    """
    with open(filename, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        # skipping header line
        next(csv_reader)
        # appending the elements as floats in the array
        for i, line in enumerate(csv_reader):
            for j in rows:
                if line[j] == "":
                    break
                X.append(fun(float(line[j])))

def process_for_prediction(img_to_test, ZZ):
    Z = []
    for csv_filename in img_to_test:
        read_rows(csv_filename, range(3), Z, lambda x: x)
        occurences = find_occurences([(Z[i], Z[i+1], Z[i+2]) for i in range(0, len(Z), 3)])
        ZZ.append(occurences)
        
