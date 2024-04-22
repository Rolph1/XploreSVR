import numpy as np
from sklearn import svm
from joblib import dump
import csv_analyzing

file_name = "img_data"
images_to_test = [22, 10, 5]
img_training = [i for i in range(15) if i not in images_to_test]
data_filename = "sample_data.csv"
X = []
Y = []

# placing the Fe2O3 concentration per image in the vector Y
# sample, siO2, Fe2O3, TiO2, Al2O3 = line[0:5]
csv_analyzing.read_rows(data_filename, [2], Y, lambda x: x)

print("finished reading chemical components")
# currently only loops through the first x images and trains the model, going to change this to take a list of images
# and then see what they give
A = []
for i in img_training:
    csv_filename = file_name + str(i)
    csv_analyzing.read_rows(csv_filename, range(3), A, lambda x: x)
    X.append(csv_analyzing.find_occurences([(A[i], A[i+1], A[i+2]) for i in range(0, len(A), 3)]))
print("finished reading RGB components")


regression_model = svm.SVR(C=1.0, kernel='rbf')

# selecting only the first x elements of Y to train the model as the rest will be used for testing
Y_cut = []
for i in img_training:
    Y_cut.append(Y[i])
print("lenths: " + str(len(X)) + " " + str(len(Y_cut)))
# for arr in X:
#     print(str(len(arr)))

# training model and placing the trained model in a joblib file to not have to train it again later on
regression_model.fit(X, Y_cut)
dump(regression_model, 'svr_model.joblib')
print("model successfully trained!")

# testing data prediction
for i in images_to_test:
    Z = []
    ZZ = []
    csv_filename = file_name + str(i)
    csv_analyzing.read_rows(csv_filename, range(3), Z, lambda x: x)
    ZZ.append(csv_analyzing.find_occurences([(Z[i], Z[i+1], Z[i+2]) for i in range(0, len(Z), 3)]))
    prediction = regression_model.predict(ZZ)
    for s in prediction:
        #print("image number " + str(s) + ":")
        print("predicted value: " + str(s))
        print("actual value: " + str(Y[i]))
        print("MSE: " + str((s-Y[i])**2))
