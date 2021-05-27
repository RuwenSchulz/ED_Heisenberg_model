import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.lib.scimath import logn
import scipy
from scipy.sparse import diags, kron, csr_matrix
from scipy import sparse, optimize
from scipy.sparse.linalg import eigs, eigsh
import datetime
import copy
from joblib import Parallel, delayed
import math
import random


import ed_1D_spinchain.variables as variables
from ed_1D_spinchain.operators import getstate,random_state
from ed_1D_spinchain.basis import createbasis

def projector_right(cutpoint):
    cut=cutpoint
    projectorlist=[[],[]]
    basis_uncut=variables.basis_index_element_map
    basis_uncut2=variables.basis_element_index_map
    spinlist_uncut=variables.spinlist
    basis_cut_left={}
    basis_cut_right={}
    len_basis_uncut=len(basis_uncut)
    
    for i in range(0,len_basis_uncut):
        basis_cut_left[i]=basis_uncut[i][:int(cut)]
        basis_cut_right[i]=basis_uncut[i][int(cut):]
        
    spinlist_left=spinlist_uncut[:int(cut)]
    createbasis(spinlist_left)
    basis_left=variables.basis_index_element_map
    basis_left_position=variables.basis_element_index_map
    spinlist_right=spinlist_uncut[int(cut):]
    createbasis(spinlist_right)
    basis_right=variables.basis_index_element_map
    basis_right_position=variables.basis_element_index_map
    data=[]
    row=[]
    col=[]
    for i in range(0,len(basis_left)):
        for j in range(len_basis_uncut):
            if((basis_left[i]+basis_cut_right[j])==basis_uncut[j]):
                data.append(1)
                row.append(basis_right_position[str(basis_cut_right[j])])
                col.append(basis_uncut2[str(basis_left[i]+basis_cut_right[j])])
 
        projectorlist[1].append(csr_matrix((data,(row,col)),shape=(len(basis_right), len_basis_uncut)))
        data=[]
        row=[]
        col=[]
    data=[0 for i in range(len(basis_right))]
    row=[i for i in range(len(basis_right))]
    col=[0 for i in range(len(basis_right))]
    for i in range(0,len(basis_right)):
        data[i]=1
        projectorlist[0].append(csr_matrix((data,(row,col)),shape=(len(basis_right),1)))
        data=[0 for i in range(len(basis_right))]
    variables.basis_index_element_map=basis_uncut
    variables.basis_element_index_map=basis_uncut2
    variables.spinlist=spinlist_uncut
    return projectorlist

def projector_left(cutpoint):
    cut=cutpoint
    projectorlist=[[],[]]
    basis_uncut=variables.basis_index_element_map
    basis_uncut2=variables.basis_element_index_map
    spinlist_uncut=variables.spinlist
    basis_cut_left={}
    basis_cut_right={}
    len_basis_uncut=len(basis_uncut)
    
    for i in range(0,len_basis_uncut):
        basis_cut_left[i]=basis_uncut[i][:int(cut)]
        basis_cut_right[i]=basis_uncut[i][int(cut):]
        
    spinlist_left=spinlist_uncut[:int(cut)]
    createbasis(spinlist_left)
    basis_left=variables.basis_index_element_map
    basis_left_position=variables.basis_element_index_map
    spinlist_right=spinlist_uncut[int(cut):]
    createbasis(spinlist_right)
    basis_right=variables.basis_index_element_map
    basis_right_position=variables.basis_element_index_map
    data=[]
    row=[]
    col=[]
    for i in range(0,len(basis_right)):
        for j in range(len_basis_uncut):
            if((basis_cut_left[j]+basis_right[i])==basis_uncut[j]):

                data.append(1)
                row.append(basis_left_position[str(basis_cut_left[j])])
          
                col.append(basis_uncut2[str(basis_cut_left[j]+basis_right[i])])
        projectorlist[1].append(csr_matrix((data,(row,col)),shape=(len(basis_left), len_basis_uncut)))
        data=[]
        row=[]
        col=[]
    data=[0 for i in range(len(basis_left))]
    row=[i for i in range(len(basis_left))]
    col=[0 for i in range(len(basis_left))]
    for i in range(0,len(basis_left)):
        data[i]=1
        projectorlist[0].append(csr_matrix((data,(row,col)),shape=(len(basis_left),1)))
        data=[0 for i in range(len(basis_left))]
    variables.basis_index_element_map=basis_uncut
    variables.basis_element_index_map=basis_uncut2
    variables.spinlist=spinlist_uncut
    return projectorlist



def densitymatrix_right(cut,state=0):
    cutpoint=cut+1
    densitymatrix=0
    projectors=[]
    Psi=0
    lamb=[]
    spinlist_total=variables.spinlist
    lenbasisstates_total=variables.lenbasisstates
    reducedbasis_index_element_map=variables.basis_index_element_map
    reducedbasis_element_index_map=variables.basis_element_index_map
    if(state=='r'):
        Psi=random_state(variables.spinlist)
    else:
        Psi=getstate(state)

    projectors=projector_right(cutpoint)
    for i in range(0,len(projectors[1])):
        projection_spin_up=scipy.sparse.csr_matrix.dot(projectors[1][i],Psi)
        densitiymatrix_element=scipy.sparse.csr_matrix.dot(projection_spin_up,projection_spin_up.transpose())
        densitymatrix+=densitiymatrix_element

    return densitymatrix


def densitymatrix_left(cut,state=0):
    cutpoint=cut+1
    densitymatrix=0
    projectors=[]
    Psi=0
    lamb=[]
    spinlist_total=variables.spinlist
    lenbasisstates_total=variables.lenbasisstates
    reducedbasis_index_element_map=variables.basis_index_element_map
    reducedbasis_element_index_map=variables.basis_element_index_map
    if(state=='r'):
        Psi=random_state(variables.spinlist)
    else:
        Psi=getstate(state)

    projectors=projector_left(cutpoint)
    for i in range(0,len(projectors[1])):
        projection_spin_up=scipy.sparse.csr_matrix.dot(projectors[1][i],Psi)
        densitiymatrix_element=scipy.sparse.csr_matrix.dot(projection_spin_up,projection_spin_up.transpose())
        densitymatrix+=densitiymatrix_element

    return densitymatrix

def densitymatix_left_and_right(cut,state=0):
    cutpoint=cut+1
    densitymatrix_left=0
    densitymatrix_right=0
    projectors=[]
    Psi=0
    lamb=[]
    spinlist_total=variables.spinlist
    lenbasisstates_total=variables.lenbasisstates
    reducedbasis_index_element_map=variables.basis_index_element_map
    reducedbasis_element_index_map=variables.basis_element_index_map
    if(state=='r'):
        Psi=random_state(variables.spinlist)
    else:
        Psi=getstate(state)

    projectors=projector_right(cutpoint)
    for i in range(0,len(projectors[1])):
        projection_spin_up=scipy.sparse.csr_matrix.dot(projectors[1][i],Psi)
        densitiymatrix_element=scipy.sparse.csr_matrix.dot(projection_spin_up,projection_spin_up.transpose())
        densitymatrix_right+=densitiymatrix_element

    projectors=projector_left(cutpoint)
    for i in range(0,len(projectors[1])):
        projection_spin_up=scipy.sparse.csr_matrix.dot(projectors[1][i],Psi)
        densitiymatrix_element=scipy.sparse.csr_matrix.dot(projection_spin_up,projection_spin_up.transpose())
        densitymatrix_left+=densitiymatrix_element
        
    return densitymatrix_left,densitymatrix_right


def proj_left_right(cut):
    cutpoint=cut+1
    projectors=projector_right(cutpoint)
    prol=projectors
    projectors=projector_left(cutpoint)
    pror=projectors

    return prol,pror

def densitymatix_AB_non_square(cut,state=0):
    cutpoint=cut+1
    right_half=0
    left_half=0
    densitymatrix_left=0
    densitymatrix_right=0
    projectors=[]
    Psi=0
    lamb=[]
    spinlist_total=variables.spinlist
    lenbasisstates_total=variables.lenbasisstates
    reducedbasis_index_element_map=variables.basis_index_element_map
    reducedbasis_element_index_map=variables.basis_element_index_map
    if(state=='r'):
        Psi=random_state(variables.spinlist)
    else:
        Psi=getstate(state)

    projectors=projector_right(cutpoint)
    prol=projectors
    #print("pror",projectors)
    for i in range(0,len(projectors[1])):
        
        projection_spin_up=scipy.sparse.csr_matrix.dot(projectors[1][i],Psi)
        
        #densitiymatrix_element=scipy.sparse.csr_matrix.dot(projection_spin_up,projection_spin_up.transpose())
        right_half+=projection_spin_up

    projectors=projector_left(cutpoint)
    pror=projectors
    #print("prol",projectors)
    for i in range(0,len(projectors[1])):
        projection_spin_up=scipy.sparse.csr_matrix.dot(projectors[1][i],Psi)
        
        #densitiymatrix_element=scipy.sparse.csr_matrix.dot(projection_spin_up,projection_spin_up.transpose())
        left_half+=projection_spin_up
    #print(left_half)
    #print(right_half)
    return scipy.sparse.csr_matrix.dot(left_half,right_half.transpose()),prol,pror


def entanglement_spectrum(cut,state=0,k=30):
    cutpoint=cut+1
    densitymatrix=0
    projectors=[]
    Psi=0
    lamb=[]
    spinlist_total=variables.spinlist
    lenbasisstates_total=variables.lenbasisstates
    reducedbasis_index_element_map=variables.basis_index_element_map
    reducedbasis_element_index_map=variables.basis_element_index_map
    if(state=='r'):
        Psi=random_state(variables.spinlist)
    else:
        Psi=getstate(state)
    projectors=projector_right(cutpoint)
    for i in range(0,len(projectors[1])):
        projection_spin_up=scipy.sparse.csr_matrix.dot(projectors[1][i],Psi)
        densitiymatrix_element=scipy.sparse.csr_matrix.dot(projection_spin_up,projection_spin_up.transpose())
        densitymatrix+=densitiymatrix_element
    density_matrix_diagonalized =scipy.sparse.linalg.eigsh(densitymatrix.toarray(),k=k,which='LA')
   # schmidt-eigenstate: density_matrix_diagonalized[1][0]
    print("trace:",sum(density_matrix_diagonalized[0]))
    density_matrix_diagonalized_real=[]
    for i in range(0,len(density_matrix_diagonalized[0])):
        density_matrix_diagonalized_real.append(density_matrix_diagonalized[0][i].real)
    density_matrix_diagonalized_real= sorted(density_matrix_diagonalized_real)
    density_matrix_diagonalized_real.reverse()
    return density_matrix_diagonalized_real,[i for i in range(0,len(density_matrix_diagonalized_real))]


def entanglement_entropy(cut,state=0):
    cutpoint=cut+1
    densitymatrix=0
    projectors=[]
    Psi=0
    lamb=[]
    spinlist_total=variables.spinlist
    lenbasisstates_total=variables.lenbasisstates
    reducedbasis_index_element_map=variables.basis_index_element_map
    reducedbasis_element_index_map=variables.basis_element_index_map
    createbasis(spinlist_total)
    unreducedbasis_index_element_map=variables.basis_index_element_map
    unreducedbasis_element_index_map=variables.basis_element_index_map
    lenunreducedbasis_index_element_map=len(unreducedbasis_index_element_map)
    if(state=='r'):
        Psi=random_state(variables.spinlist)
    else:
        Psi=getstate(state)
    psiarraydata=[i[0] for i in Psi.toarray()]
    row=[]
    for j in range(lenbasisstates_total):
        row.append(unreducedbasis_element_index_map[str(reducedbasis_index_element_map[j])])
    Psi_in_unreduced_basis=csr_matrix((psiarraydata,(row,np.zeros(lenbasisstates_total))),shape=(lenunreducedbasis_index_element_map, 1))
    projectors=projector_right(cutpoint)
    for i in range(0,len(projectors[1])):
        projection_spin_up=scipy.sparse.csr_matrix.dot(projectors[1][i],Psi_in_unreduced_basis)
        densitiymatrix_element=scipy.sparse.csr_matrix.dot(projection_spin_up,projection_spin_up.transpose())
        densitymatrix+=densitiymatrix_element
    S=0
    entropyhalfwaydone=scipy.sparse.csr_matrix.dot(densitymatrix,scipy.linalg.logm(densitymatrix.toarray()))
    for i in range(0,len(projectors[0])):
       S+=scipy.sparse.csr_matrix.dot(projectors[0][i].transpose(),scipy.sparse.csr_matrix.dot(entropyhalfwaydone,projectors[0][i]))
    return -S[0][0]
