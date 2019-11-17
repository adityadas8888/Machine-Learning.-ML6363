import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

H=[0.1,1,5,10]
for a in range(len(H)):
    h=H[a]
    kh=0
    px=0

    mean1 = [1,0]
    mean2 = [0,1.5]
    Sigma1 = [[0.9,0.4],[0.4,0.9] ]
    Sigma2 = [[0.9,0.4],[0.4,0.9] ]
    
    x1, y1 = np.random.multivariate_normal(mean1, Sigma1, 300).T
    x2, y2 = np.random.multivariate_normal(mean2, Sigma2, 300).T
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    X = np.arange(min(x),max(x) , 0.1)
    Y = np.arange(min(y),max(y) , 0.1)
    
    l1=len(x)
    pxy = []

    kh = np.zeros(X.size)

    for j in range(len(X)):
        for k in range(len(Y)):
            kh = np.absolute( ((X[j] - x)/h) * ((Y[k] - y)/h) )
            K = (np.where(kh <= 0.5, 1.0, 0.0))
            p_xy = (np.sum(K))/(l1*(h**2))
            pxy.append([X[j],Y[k],p_xy])

    pxy = np.array(pxy)

    ax = plt.axes(projection='3d')
    ax.plot_trisurf(pxy[:,0], pxy[:,1], pxy[:,2], cmap='viridis', edgecolor='none')
    plt.show()