def transpositionMatrix(n,r1,r2):
    """
    This function creates a row permuation matrix P. 

    Input: size n, index of rows r1, r2 to be permuted
    """
    P=np.eye(n); P[r1,r2]=1; P[r2,r1]=1; P[r1,r1]=0; P[r2,r2]=0;
    return P

def forwardSolve(L,B):
    """
    This code implements a forward substitution solver

    Input: Lower diagonal non-singular matrix L, matrix B
    Output: Matrix X such that LX=B
    """
    n=L.shape[0]
    if len(B.shape)<=1:
        m=1
    else:
        m=B.shape[1]

    if B.shape[0]!=n:
        raise ValueError("Incompatible sizes. L has size {} whereas B has size {}.".format(L.shape,B.shape))

    X=np.zeros([n,m])

    for i in range(n):
        if L[i,i]==0:
            raise ValueError("Matrix should be non-singular")
        X[i]=(B[i]-sum([L[i,t]*X[t] for t in range(i)]))/L[i,i]

    return X

def backwardSolve(U,B):
    """
    This code implements a backward substitution solver

    Input: Lower diagonal non-singular matrix L, matrix B
    Output: Matrix X such that LX=B
    """
    n=U.shape[0]
    if len(B.shape)<=1:
        m=1
    else:
        m=B.shape[1]

    if B.shape[0]!=n:
        raise ValueError("Incompatible sizes. L has size {} whereas B has size {}.".format(L.shape,B.shape))

    X=np.zeros([n,m])

    for i in range(n):
        i=n-1-i
        if U[i,i]==0:
            raise ValueError("Matrix should be non-singular")
        X[i]=(B[i]-sum([U[i,t]*X[t] for t in range(i+1,n)]))/U[i,i]

    return X


def LU(A):
    """
    This function implements simple LU decomposition heuristic using Gaussian elimination.
    The complexity of the algorithm is O(n^3/2). WARNING: Not vectorized, so it might be slow for large matrices.
    """
    A=A.astype('f')
    n=A.shape[0]
    L=np.eye(n)
    U=A.copy()
    P=np.eye(n)
    Pinv=np.eye(n)
    for i in range(n):
        stepDone=False
        while  not stepDone: 
            if U[i,i]!=0:
                #Gaussian Elimination Step
                tempL=np.eye(n)
                for i2 in range(i+1,n):
                    l=U[i2,i]/U[i,i]
                    U[i2]=U[i2]-l*U[i]
                    tempL[i2,i]=-l
                Lseq.append(tempL)
                L=np.matmul(tempL,L)
                stepDone=True                   
            else:
                #Check non-zero element in column
                i2=i+1
                while i2<n:
                    if U[i2,i]!=0:
                        #Permute rows
                        temp=U[i].copy()
                        U[i]=U[i2]
                        U[i2]=temp
                        #Save Permutation Matrix
                        P=np.matmul(transpositionMatrix(n,i2,i),P)
                        Pinv=np.matmul(Pinv,transpositionMatrix(n,i2,i))
                        break
                    else:
                        i2+=1
                if i2>=n:
                    #No non-zero value found, continue to next step
                    stepDone=True

    L=forwardSolve(L,np.eye(n))    
    return P, Pinv, L, U 

def LUSolver(A,B):
    """
    This program solves the system of linear AX=B equation using LU factorization
    
    Input: Matrices A and B of compatible sizes
    """
    P,Pinv, L,U=LU(A)
    X=forwardSolve(L,np.matmul(Pinv,B))
    X=backwardSolve(U,X)
    
    return X

def Cholesky(A):
    """
    This is a simple algorithm to calculate the cholesky factorization of a
    positive definite matrix. WARNING: Precision is not the best for this one.
    
    Input: Square, positive definite A (no complex for now)
    Output: Lower triangular matrix L and diagonal matrix D such that A=LDL^T
    """
    A=A.astype('f')
    n=A.shape[0]
    
    #Sanity check
    if n!=A.shape[1]:
        raise ValueException("Matrix A should be square matrix")
    if not (A==A.T).all():
        raise ValueException("Matrix A should symmetric")
    
    i=0
    L=np.eye(n)
    while i<n:
        a=A[i,i]
        if a==0:
            raise ValueException("Matrix A is singular.")
        if a<0:
            raise ValueException("Matrix is not positive semidefinite.")
            
        b=A[i+1:,i].reshape((n-i-1,1))
        B=A[i+1:,i+1:]-np.matmul(b,b.T)/a
        
        #Update A
        subA=np.block([[A[:i, :i], np.zeros((A[:i,:i].shape[0],1))],[np.zeros((1,A[:i,:i].shape[1])),a]])
        A=np.block([[subA, np.zeros((subA.shape[0],n-subA.shape[0]))],[np.zeros((n-subA.shape[0],subA.shape[1])), B]])
        
        #Update L
        subL=np.block([[1, np.zeros((1,b.shape[0]))],[b/a,np.eye(b.shape[0]) ]])
        tempL=np.block([[np.eye(i), np.zeros((i,subL.shape[0]))], [np.zeros((subL.shape[0],i)),subL]])
        L=np.matmul(L,tempL)
        i+=1
    return L, A