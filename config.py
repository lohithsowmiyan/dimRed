import argparse

def parse_arguments():

    parser = argparse.ArgumentParser(description = "dimensionality reduction for SE data")
    parser.add_argument('-d', '--dataset', type = str, default= 'data/optimize/config/SS-L.csv')
    parser.add_argument('-e', '--epochs', type = int, default = 100)
    parser.add_argument('-l', '--loss', type = str, default = 'mse')
    parser.add_argument('-o', '--optimizer', type = str, default = 'adam')

    args = parser.parse_args()

    return args