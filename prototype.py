from scipy.optimize import minimize
import numpy as np

def obj(X,*args):
    X = X.reshape((int(len(X)/2),2))
    xf = args[0][0]
    dist_sum = 0
    for i in range(np.size(X,0)):
        dist_sum += np.linalg.norm(X[i,:] - xf)
    return dist_sum

def anchor_con(X,*args):
    X = X.reshape((int(len(X)/2),2))
    x0, step_size = args[0][0], args[0][1]
    constraint = []
    for i in range(np.size(X,0)):
        if i == 0:
            constraint.append(np.linalg.norm(X[i,:]-x0)-step_size)
        else:
            constraint.append(np.linalg.norm(X[i,:]-X[i-1,:])-step_size)
    print(constraint)
    return np.array(constraint)

if __name__ == '__main__':
    X = np.array([
        [1,1],
        [2,2],
        [3,3]
    ])

    xf = np.array([10,10])
    args = [xf]

    x0 = np.array([0,0])
    step_size = 1.0
    con_args = [x0,step_size]

    con = (
        {'type':'eq',
        'fun':anchor_con,
        'args':con_args}
    )

    x = minimize(obj,X,args=args,constraints=con)
    
    print(x)

