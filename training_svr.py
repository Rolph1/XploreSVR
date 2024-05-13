import numpy as np
from joblib import dump
import csv_analyzing
from lssvr import LSSVR 
file_name = "img_data"
images_to_test = [5, 10, 19] #, 12, 11, 24
img_training = [i for i in range(3) if i not in images_to_test]
data_filename = "sample_data.csv"
X = []
Y = []
model = LSSVR(kernel='rbf', gamma=0.01)
# placing the Fe2O3 concentration per image in the vector Y
# sample, siO2, Fe2O3, TiO2, Al2O3 = line[0:5]
csv_analyzing.read_rows(data_filename, [4], Y, lambda x: x)

print("finished reading chemical components")
# currently only loops through the first x images and trains the model, going to change this to take a list of images
# and then see what they give
A = []
for i in img_training:
    csv_filename = file_name + str(i)
    csv_analyzing.read_rows(csv_filename, range(3), A, lambda x: x)
    split_A = [(A[i], A[i+1], A[i+2]) for i in range(0, len(A), 3)]
    # X.append(csv_analyzing.calculate_features(split_A))
    occ = csv_analyzing.find_occurences(split_A)
    X.append(occ)
    print(len(X))
    print(len(occ))
    
    #print(np.shape(np.array(X)))
print("finished reading RGB components")


regression_model = LSSVR(kernel='rbf', gamma=0.01)

# selecting only the first x elements of Y to train the model as the rest will be used for testing
Y_cut = []
for i in img_training:
    Y_cut.append(Y[i])
print("lenths: " + str(len(X)) + " " + str(len(Y_cut)))
print("Y is " + str(Y))
# for arr in X:
#     print(str(len(arr)))

#X_processed = csv_analyzing.preprocessing(X)
#print(np.shape(np.array(X_processed)))
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
    
    # ZZ.append(csv_analyzing.calculate_features(Z))
print("prediction result on test data " + str(np.shape(np.array(ZZ))))
prediction = []
for i in ZZ:
     prediction.append(regression_model.predict(np.reshape(np.array(i), (-1, 1))))
for i, s in enumerate(prediction):
        #print("image number " + str(s) + ":")
        print("predicted value: " + str(s))
        print("actual value: " + str(Y[i]))
        print("MSE: " + str((s-Y[i])**2))