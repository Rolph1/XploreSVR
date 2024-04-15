import csv


def normalize_rgb(x):
    return int(x)/255


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