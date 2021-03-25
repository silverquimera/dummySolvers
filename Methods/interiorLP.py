import numpy as np

def diag2(x):
    """
    Return diagonal matrix diag(x). Different to np.diag as np.diag of the zero vector returns a 1x1 matrix always.
    """
    n=len(x)
    X=np.zeros((n,n))
    for i in range(n):
        X[i,i]=x[i]
        
    return X
    

def basicInteriorLP(c,A,b):
    """
    This is a basic solver of LP using interior point method algorithm.
    
    Input: Problem in form min c^t x : A x <= b
    Output: Optimal Value, Primal Solution, Dual Solution
    
    WARNING: Not efficient for the moment... Mehrotra's simple implementation in progress
    """
    epsilon=1e-10
    (m,n)=A.shape
    
    x=np.ones((n,1))
    s=np.ones((n,1))
    l=np.ones((m,1))
    
    #duality measure
    mu=np.mean(x*s)
    err1=np.linalg.norm(b-np.matmul(A,x))/max([np.linalg.norm(b),1])
    err2=np.linalg.norm(c-s-np.matmul(A.T,l))/max([np.linalg.norm(c),1])

    while max([mu,err1,err2])>epsilon:
        err1=np.linalg.norm(b-np.matmul(A,x))/max([np.linalg.norm(b),1])
        err2=np.linalg.norm(c-s-np.matmul(A.T,l))/max([np.linalg.norm(c),1])
        
        print("Central: {} Primal: {} Dual: {}".format(mu,err1,err2))
        #Compute the jacobian and function:
        J=np.block([[np.zeros((n,n)),A.T,np.eye(n)],[A,np.zeros((m,m)),np.zeros((m,n))],[diag2(s),np.zeros((n,m)), diag2(x)]])
        F=np.block([[np.matmul(A.T,l)+s-c],[np.matmul(A,x)-b],[x*s]])
        
        #Solve the Newton step:
        
        #Find an LU factorization of the jacobian
        P,Pinv,L,U=LU(J)
        
        #Find the affine scaling direction:
        aff=forwardSolve(L,np.matmul(Pinv,-F))
        aff=backwardSolve(U,aff)
        
        aff=np.linalg.solve(J,-F)
        
        #Determine if step is good
        dx=aff[:n]
        ds=aff[-n:]
        dl=aff[n:-n]
        delta1=min([1]+[-x[i]/dx[i] for i in range(n) if dx[i]<0])
        delta2=min([1]+[-s[i]/ds[i] for i in range(n) if ds[i]<0])
        
        if min([delta1,delta2])<1:
            #not full step possible, we need center-correction
            
            muaff=np.mean((x+delta1*dx)*(s+delta2*ds))
            sigma=min(0.208, (muaff/mu)**2)
            
            F2=np.block([[np.zeros((n+m,1))],[aff[:n]*aff[-n:]-sigma*muaff*np.ones((n,1))]])
            
            #ccorr=forwardSolve(L,np.matmul(Pinv,-F2))
            #ccorr=backwardSolve(U,ccorr)
            
            ccorr=np.linalg.solve(J,-F2)
            
            dx+=ccorr[:n]
            dl+=ccorr[n:-n]
            ds+=ccorr[-n:]
        
        #step length to guarantee feasibility
        delta1=min([1]+[-x[i]/dx[i] for i in range(n) if dx[i]<0])
        delta2=min([1]+[-s[i]/ds[i] for i in range(n) if ds[i]<0])
        
        #step length to guarantee that we are close to the central path:
        
        for alpha in [1, .9975, .95, .90, .75, .50]:
            xs=(x+alpha*delta1*dx)*(s+alpha*delta2*ds)
            if np.min(xs)>=1e-6*np.mean(xs):
                delta1=alpha*delta1
                delta2=alpha*delta2
                break
        
        x+=delta1*dx
        s+=delta2*ds
        l+=delta2*dl
        
        mu=np.mean(x*s)
    return np.matmul(c.T,x), x,s  