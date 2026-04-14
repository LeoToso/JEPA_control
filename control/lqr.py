"""Discrete-time LQR solver via DARE."""
from __future__ import annotations
from typing import Tuple
import numpy as np
import scipy.linalg

def solve_discrete_lqr(A,B,Q,R):
    try:
        P=scipy.linalg.solve_discrete_are(A,B,Q,R)
    except Exception:
        P=_dare_iteration(A,B,Q,R)
    K=np.linalg.solve(R+B.T@P@B,B.T@P@A)
    A_cl=A-B@K
    eigs=scipy.linalg.eigvals(A_cl)
    return K,P,eigs

def _dare_iteration(A,B,Q,R,max_iter=1000,tol=1e-10):
    P=Q.copy()
    for _ in range(max_iter):
        P_new=Q+A.T@P@A-A.T@P@B@np.linalg.solve(R+B.T@P@B,B.T@P@A)
        if np.linalg.norm(P_new-P,'fro')<tol:
            return P_new
        P=P_new
    return P
