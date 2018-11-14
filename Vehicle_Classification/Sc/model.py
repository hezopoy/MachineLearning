import numpy as np
import math
def Gaussian_ML_estimate(X):
      #[l,N]=size(X);
      #l colls (classdata)
      #N rows (features)
      N=np.size(X[0,:]) #3500
      l =np.size(X[:,0])
 #     print l
#      print N
      m_hat=np.zeros((1,N))
      for x in range(N):
            m_hat[:,x]= sum(X[:,x])
      m_hat=(1/float(l))* m_hat.T
      S_hat=np.zeros(N)
      #print X[:,1]
      for k in range(l):
            a = np.subtract(X[k,:],m_hat.T)
            b = np.matmul(a.T,a)
            S_hat=np.add(S_hat,b)
      S_hat=(1/float(l))*S_hat
      return [m_hat,S_hat]
"""
X = np.array([[1,2,1,4],[2,2,5,7]]) 
[m,S] = Gaussian_ML_estimate(X)
print m
print S
"""
def euclidean_classifier(m,X):
      #[l,c]=size(m);(m.T)
      c=np.size(m[:,0])
      #c (classdata)
      #[l,N]=size(X);
      #l colls (classdata)
      #N rows (features)
      #l=np.size(X[0,:])
      N =np.size(X[:,0])
      #print l
      #print N
      de = range(c)
      #z = range(l)
      for j in range(c):
            sub = np.subtract(X[0,:],m[j,:].T)
            mul = np.matmul(sub,sub.T)                  
            de[j]=math.sqrt(mul)
      return [np.argmin(de),min(de)]
