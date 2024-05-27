import numpy as np
from joblib import dump
import csv_analyzing
from lssvr import LSSVR 
from sklearn.svm import SVR
file_name = "img_data"
images_to_test = [9] #, 12, 11, 24    ; 5,10,19
images_to_test2 = [38]
img_training = [i for i in range(50) if i not in images_to_test and i not in images_to_test2]
data_filename = "sample_data.csv"
X = []
Y = []
model = LSSVR(kernel='rbf', gamma=0.01)
# placing the Fe2O3 concentration per image in the vector Y
# sample, siO2, Fe2O3, TiO2, Al2O3 = line[0:5]
csv_analyzing.read_rows(data_filename, [4], Y, lambda x: x)

print("finished reading chemical components")

#pooling functions (not used currently as they lower accuracy)
def pool_block(block):
    return np.mean(block, axis=0)

# Function to reshape and pool the data
def pool_data(A):
    pooled_A = []
    A = np.array(A).reshape(-1, 3)
    for i in range(0, len(A) - 1, 2):
        if i % 2 == 0 and (i + 3) < len(A):
            block = A[i:i + 4]
            pooled_A.append(pool_block(block))
        else:
            pooled_A.append(A[i])
    return pooled_A

# Read and process training data with pooling
for i in img_training:
    csv_filename = file_name + str(i)
    A = []
    csv_analyzing.read_rows(csv_filename, range(3), A, lambda x: x)
    split_A = [(A[i], A[i+1], A[i+2]) for i in range(0, len(A), 3)]
    #pooled_A = pool_data(split_A)  uncomment to use pooling
    #occ = csv_analyzing.find_occurences(pooled_A)
    occ = csv_analyzing.find_occurences(split_A)
    X.append(occ)
    print("Length of X:", len(X))
    print("Length of occurrences:", len(occ))

print("Finished reading RGB components")

regression_model = SVR(kernel='rbf', C=30.0, epsilon=0.01, gamma='scale')

# selecting only the first x elements of Y to train the model as the rest will be used for testing
Y_cut = []
for i in img_training:
    Y_cut.append(Y[i])
print("lenths: " + str(len(X)) + " " + str(len(Y_cut)))
print("Y is " + str(Y))

# training model and placing the trained model in a joblib file to not have to train it again later on
regression_model.fit(X, Y_cut)
dump(regression_model, 'svr_model.joblib')

print("model successfully trained!")
Z = []
ZZ = []
# testing data prediction

for i in images_to_test:
    csv_filename = file_name + str(i)
    csv_analyzing.read_rows(csv_filename, range(3), Z, lambda x: x)
    occ = csv_analyzing.find_occurences([(Z[i], Z[i+1], Z[i+2]) for i in range(0, len(Z), 3)])
    ZZ.append(occ)
    print(len(occ))
    print(len(ZZ))

K = []
KK = []

for i in images_to_test2:
    csv_filename = file_name + str(i)
    csv_analyzing.read_rows(csv_filename, range(3), K, lambda x: x)
    occ = csv_analyzing.find_occurences([(K[i], K[i+1], K[i+2]) for i in range(0, len(K), 3)])
    KK.append(occ)
    print(len(occ))
    print(len(KK))
    # ZZ.append(csv_analyzing.calculate_features(Z))
    
print("prediction result on test data " + str(np.shape(np.array(ZZ))))
prediction = regression_model.predict(ZZ)
for i, s in enumerate(prediction):
        #print("image number " + str(s) + ":")
        print("predicted value: " + str(s))
        print("actual value: " + str(Y[images_to_test[i]]))
        print("MSE: " + str((s-Y[images_to_test[i]])**2))
        
prediction = regression_model.predict(KK)
for i, s in enumerate(prediction):
        #print("image number " + str(s) + ":")
        print("predicted value: " + str(s))
        print("actual value: " + str(Y[images_to_test2[i]]))
        print("MSE: " + str((s-Y[images_to_test2[i]])**2))
