# dummySolvers
Basic implementations of linear algebra algorithms and solvers. Currently, the package contain basic implementations of LU, LDL decompositions along with its respective solvers for linear equations (forward/barckward substitution solvers). We also have a simple LP solver using a interior point method (dummified version of Mehrotra's method). Here are some examples (for more, check out the notebook).

## LU factorization

```python
>>A=np.array([[1,4,1,0],[1,1,0,1]])
>>P,Pinv,L,U=LU(A)
>>(L,U)
>>(array([[1., 0.],
        [1., 1.]]),
 array([[ 1.,  4.,  1.,  0.],
        [ 0., -3., -1.,  1.]], dtype=float32))
```

## LU linear solver
```python
>>b=np.array([[8],[4]])
>>x=LUSolver(A,b)
>>array([[2.66666667],
       [1.33333333]])
```

## Cholesky (LDL) factorization

```python
>>B=np.matmul(A,A.T)
>>L,D=cholesky(B)
>>(L,D)
>>(array([[1.        , 0.        ],
        [0.27777779, 1.        ]]),
 array([[18.        ,  0.        ],
        [ 0.        ,  1.61111116]]))
 ```
 
 ## LP solver
 ```python
 >> c=np.array([[2],[5],[0],[0]])
 >> basicInteriorLP(-c,A,b)
 >> Central: 1.0 Primal: 0.25 Dual: 2.3044185443591205
Central: 0.2475916448278927 Primal: 0.03850594572783072 Dual: 0.35202986224744426
Central: 0.01897211394479829 Primal: 9.626486431957293e-05 Dual: 0.015113703207410645
Central: 4.967101781870736e-05 Primal: 2.4917644372290935e-07 Dual: 3.7784258018603556e-05
Central: 1.2417928376690224e-07 Primal: 6.229414539760129e-10 Dual: 9.446064510742916e-08
Central: 3.1044822002444885e-10 Primal: 1.557309836641707e-12 Dual: 2.3615156258039155e-10
Central: 7.761205501273799e-13 Primal: 3.9968028886505635e-15 Dual: 5.903733510635419e-13
>>(array([[-12.]]),
 array([[2.66666667e+00],
        [1.33333333e+00],
        [2.74068080e-16],
        [6.46955452e-15]]),
 array([[3.09190749e-16],
        [1.44805677e-16],
        [1.00000000e+00],
        [1.00000000e+00]]))
 ```
