import numpy as np
from pylab import *
#the number of combined gaussian distribution
K = 2
# scaling the data
def scaling(X):
    clm = X.shape[1]

    average = np.mean(X,axis = 0)
    sigma = np.std(X,axis=0)

    for i in range(clm):
        X[:,i] =(X[:,i] - average[i])/sigma[i]

    return X

def gaussian(x,average,cov):
    temp1 = 1/((2*np.pi)**(x.size/2.0))
    temp2 = 1/(np.linalg.det(cov)**0.5)
    temp3 = -0.5 * np.dot(np.dot(x-average,np.linalg.inv(cov)),x-average)

    return temp1 * temp2 * np.exp(temp3)

def likelihood(X,average,cov,pi):

    sum = 0.0
    for n in range(len(X)):
        temp = 0.0
        for k in range(K):
            temp += pi[k] * gaussian(X[n],average[k],cov[k])
        sum += np.log(temp)
    return sum

if __name__ == '__main__':
    data = np.genfromtxt('faithful.txt')
    X = data[:,0:2]
    print X
    X = scaling(X)
    length = len(X)

    average = np.random.rand(K,2)
    cov = zeros((K,2,2))
    for k in range(K):
        cov[k] = [[1.0,0.0],[0.0,1.0]]
    pi = np.random.rand(K)
    gamma = zeros((length,K))

    like = likelihood(X,average,cov,pi)
    like_store = np.ones(0)
    
    turn = 0
     
    while True:
        print turn, like
        # e step

        for n in range(length):
            denominator = 0.0
            for j in range(K):
                denominator += pi[j] * gaussian(X[n],average[j],cov[j])
            for k in range(K):
                gamma[n][k] = pi[k] * gaussian(X[n],average[k],cov[k])/denominator

        for k in range(K):
            Nk = 0.0
            for n in range(length):
                Nk += gamma[n][k]
            average[k] = array([0.0,0.0])

            for n in range(length):
                average[k] += gamma[n][k] * X[n]
            average[k] /= Nk

            cov[k] = array([[0.0,0.0], [0.0,0.0]])
            for n in range(length):
                temp = X[n] - average[k]
                cov[k] += gamma[n][k] * matrix(temp).reshape(2,1)*matrix(temp).reshape(1, 2) 
            cov[k] /= Nk

            pi[k] = Nk/length
        print 'debug4'
        print X.shape,average.shape,cov.shape,pi.shape
        print cov
        new_like = likelihood(X,average,cov,pi)
        print 'debug5'
        diff = new_like - like
        # it is the condition of finishing update
        if diff < 0.01:
            break
        like_store = np.append(like_store,like)
        like = new_like
        turn += 1


# dipicting a graph
xlist = np.linspace(-2.5,2.5,50)
ylist = np.linspace(-2.5,2.5,50)
like_x = np.arange(like_store.size)
plt.plot(like_store,'.r')
plt.xlabel('number of iteration')
plt.ylabel('likelihood of log')
xlim(0,like_store.size)
plt.savefig('em.png')
plt.show()
x,y = np.meshgrid(xlist,ylist)
'''for k in range(K):
    z = bivariate_normal(x,y,np.sqrt(cov[k,0,0]),np.sqrt(cov[k,1,1]),average[k,0],average[k,1],cov[k,0,1])
    if k == 0:
        cs = contour(x,y,z,3,colors = 'k',linewidths=1)
    else :
        cs = contour(x,y,z,3,colors = 'g',linewidths=1)
plot(X[:,0],X[:,1],'bx')

xlim(-2.5,2.5)
ylim(-2.5,2.5)
savefig('emalgorithm.png')
show()'''
