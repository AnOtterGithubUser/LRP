import numpy as np


def MNISTData(path=''):

	fx = '%s/t10k-images.idx3-ubyte' % path
	ft = '%s/t10k-labels.idx1-ubyte' % path

	X = np.fromfile(open(fx),dtype='ubyte',count=16+784*10000)[16:].reshape([10000, 784])
	T = np.fromfile(open(ft),dtype='ubyte',count=8+10000)[8:]
	T = (T[:, np.newaxis] == np.arange(10))*1.0

	return X/255.0, T
