"""Luenberger observer design."""
from __future__ import annotations
import warnings
from typing import Optional
import numpy as np
import scipy.linalg
import scipy.signal

def design_luenberger(A,B,C,pole_scale=0.8,controller_poles=None):
    n=A.shape[0]
    if controller_poles is None:
        eigs_A=scipy.linalg.eigvals(A)
        desired_poles=pole_scale*eigs_A
    else:
        desired_poles=pole_scale*controller_poles
    abs_poles=np.abs(desired_poles)
    desired_poles=np.where(abs_poles>=1.0,desired_poles/(abs_poles+1e-10)*0.9,desired_poles)
    try:
        result=scipy.signal.place_poles(A.T,C.T,desired_poles)
        L=result.gain_matrix.T
    except Exception as exc:
        warnings.warn(f'Pole placement failed: {exc}. Using DARE-based observer.')
        L=_dare_observer(A,C)
    return L

def _dare_observer(A,C):
    n=A.shape[0]
    p=C.shape[0]
    Q_w=np.eye(n)
    R_v=np.eye(p)
    try:
        P=scipy.linalg.solve_discrete_are(A.T,C.T,Q_w,R_v)
        L=P@C.T@np.linalg.solve(C@P@C.T+R_v,np.eye(p))
    except Exception:
        L=np.zeros((n,p))
    return L

class LuenbergerObserver:
    def __init__(self,A,B,C,L):
        self.A=A
        self.B=B
        self.C=C
        self.L=L
        self.n=A.shape[0]
        self.state=np.zeros(self.n)
    def reset(self,z0=None):
        self.state=z0.copy() if z0 is not None else np.zeros(self.n)
    def update(self,y,a):
        z_hat=self.state.copy()
        innovation=y-self.C@z_hat
        self.state=self.A@z_hat+self.B@a+self.L@innovation
        return z_hat

