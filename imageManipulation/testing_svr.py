from joblib import load
import csv_analyzing

# Loading model from training
regr = load('svr_model.joblib')
Y = []
data_filename = "sample_data.csv"
csv_analyzing.read_rows(data_filename, [2], Y, lambda x: x)

images_to_test = [8, 9, 10, 11]
data_filename = "sample_data.csv"


print("finished reading chemical components")