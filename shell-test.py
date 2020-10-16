import sys
import numpy
import pandas

def main(model_name, signal_filename):
    model_path = 'LightGBM-models/{0}.txt'.format(model_name)
    # print(model_path)
    signal = pandas.read_csv(signal_filename)
    signal = signal.to_numpy()
    if signal.ndim==1: # If it's a single row, for some reason (like a for loop through a pandas DataFrame):
        signal = signal.reshape(1,9)
    if signal.shape[1]!=9: # If the signal doesn't include the necessary items for the model to predict
        raise ValueError("{} signal to numpy is not in the shape of (-1, 9)".format(sys.argv[2]))
    # print(signal)
    prediction = [0 for i in signal]
    return prediction

if __name__ == '__main__':
    model_name = sys.argv[1]
    signal_filename = signal_filename = sys.argv[2]
    prediction = main(model_name,signal_filename)
    prediction = str(prediction)
    sys.stdout.write(prediction)
    sys.exit(0)
