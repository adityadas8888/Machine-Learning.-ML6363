import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mean1 = [5]
Sigma1 = [1]
Sigma2 = [0.2]
mean2 = [0]
H=[0.1,1,5,10]
for a in range(len(H)):
    h=H[a]
    kh=0
    px=0
    x = Sigma1 * np.random.randn(500) + mean1
    y = Sigma2 * np.random.randn(500) + mean2                       # 2nd set of Gaussian Data.
    x = np.concatenate((x, y))                                      # Concatenating the 2 sets of Gaussian Data.
    X = np.arange(min(x),max(x) , 0.001)

    px=[]
    l1=len(x)

    for j in range(len(X)):
        kh = np.absolute((X[j]-x)/(h))    
        k = np.where(kh <= 0.5, 1, 0)
        p_x = float(np.sum(k)/h)/l1
        px.append(p_x)
    plt.hist(x, normed=True, bins=30)
    plt.plot(X,px)
    plt.show()
