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
from scipy.linalg import expm
import cmath
import ed_1D_spinchain.variables as variables
import ed_1D_spinchain.basis as basis

def random_state(spinlist_input):
    len_state=int((2*spinlist_input[0]+1)**len(spinlist_input))
    row = np.arange(len_state)
    col = np.zeros(len_state)
    da=[]
    for i in range(0,len_state):
        da.append(random.random())
    data= np.array(da)
    #data = np.random.rand(1,len_state)[0]
    state = csr_matrix((data,(row,col)),shape=(len_state, 1))
    norm_factor=scipy.sparse.csr_matrix.dot(state.transpose(),state).toarray()[0][0]
    norm_factor=1/np.sqrt(norm_factor)
    state= scipy.sparse.csr_matrix.dot(norm_factor,state)
    return state

def norm_state(state):
    norm_factor=scipy.sparse.csr_matrix.dot(state.transpose(),state).toarray()[0][0]
    norm_factor=1/np.sqrt(norm_factor)
    return scipy.sparse.csr_matrix.dot(norm_factor,state)
    
def real_state(state):
    state_real=[]
    for i in state.toarray():
        #print(i)
        state_real.append([i[0].real])
    return sparse.csr_matrix(state_real) 

def m_s_state_i(i,j):
    var6=variables.basis_index_element_map[i]
    m_s_i_j=int(var6[j])-(variables.spinlist[j])
    return m_s_i_j

def identity(kk):
    return scipy.sparse.identity(kk)

def ladder_factor_up(i,k):
        return np.sqrt(variables.spinlist[k]*(variables.spinlist[k]+1)-m_s_state_i(i,k)*(m_s_state_i(i,k)+1))

def ladder_factor_down(i,k):
        return np.sqrt(variables.spinlist[k]*(variables.spinlist[k]+1)-m_s_state_i(i,k)*(m_s_state_i(i,k)-1))

def getstate(j):
    state_i=[]
    for i in range(0,len(variables.states)):
        state_i.append(variables.states[i][j])
    return csr_matrix(state_i).transpose()

def S_squared_i(i):
     return variables.spinlist[i]*(variables.spinlist[i]+1)*identity(variables.lenbasisstates)

def S_up_i_a(i): 
    row=np.zeros(variables.lenbasisstates)
    col=np.zeros(variables.lenbasisstates)
    data=np.zeros(variables.lenbasisstates)
    for j in range(0,variables.lenbasisstates):
        state=variables.basis_index_element_map[j]
        k=m_s_state_i(j,i)
        if (k == variables.spinlist[i]):
            continue
        else:
            state_up = state[:]
            state_up[i] = state[i] + 1
            p=variables.basis_element_index_map[str(state_up)]
            data[j]= ladder_factor_up(j,i)
            row[j]=j
            col[j]=p
    return csr_matrix((data,(row,col)),shape=(variables.lenbasisstates, variables.lenbasisstates))

def S_down_up_i(i):    
    row=np.zeros(variables.lenbasisstates)
    col=np.zeros(variables.lenbasisstates)
    data=np.zeros(variables.lenbasisstates)
    for j in range(0,variables.lenbasisstates):
        state=variables.basis_index_element_map[j]
        k=m_s_state_i(j,i%(variables.L))
        l=m_s_state_i(j,(i+1)%(variables.L))
        if (k == variables.spinlist[i%(variables.L)]):
                continue
        if (l == -variables.spinlist[(i+1)%(variables.L)]):
                continue
        else:
            data[j]= ladder_factor_down(j,(i+1)%(variables.L))*ladder_factor_up(j,i%(variables.L))
            state_up = state[:]
            state_up[i%(variables.L)] = state[i%(variables.L)] + 1
            state_up[(i+1)%(variables.L)] = state_up[(i+1)%(variables.L)] - 1
            
            p=variables.basis_element_index_map[str(state_up)]
            
            row[j]=j
            col[j]=p         
    return csr_matrix((data,(row,col)),shape=(variables.lenbasisstates, variables.lenbasisstates))

def S_z_i(i):
    row=np.zeros(variables.lenbasisstates)
    col=np.zeros(variables.lenbasisstates)
    data=np.zeros(variables.lenbasisstates)
    for j in range(0,(variables.lenbasisstates)):
        row[j]=j
        col[j]=j
        data[j]=m_s_state_i(j,i)
    return csr_matrix((data,(row,col)),shape=(variables.lenbasisstates, variables.lenbasisstates))

def time_reversal():
    a=expm(-1*1j*cmath.pi*S_y_i(0))
    for i in range(0,variables.L):
        a=a.dot(expm(-1j*cmath.pi*S_y_i(i)))
    return a

def rotation_Z2_Z2(alpha):
    R=0
    a=S_x_i(0)
    for i in range(1,variables.L):
        a+=S_x_i(i)
    b=S_z_i(0)
    for i in range(1,variables.L):
        b+=S_z_i(i)
    return expm(-1j*cmath.pi*(a)*alpha),expm(-1j*cmath.pi*(b)*alpha)

def rotation_z(alpha):
    a=expm(1j*cmath.pi*(S_z_i(0))*alpha)
    for i in range(1,variables.L):
        a=a.dot(expm(1j*cmath.pi*(S_z_i(i))*alpha))
    return a

def rotation_z_i(alpha,i):
    R=expm(1j*cmath.pi*S_z_i(i)*alpha)
    return R

def rotation_x_i(alpha,i):
    R=expm(1j*cmath.pi*S_x_i(i)*alpha)
    return R


def rotation_x(alpha):
    a=expm(1j*cmath.pi*(S_x_i(0))*alpha)
    for i in range(1,variables.L):
        a=a.dot(expm(1j*cmath.pi*(S_x_i(i))*alpha))
    return a

#S=1/2
def S_x():
    row= np.array([1,0])
    col= np.array([0,1])
    data=(1/2)*np.array([1,1])
    return csr_matrix((data,(row,col)), shape=(2,2))

def S_x_i_half(j):
    var4=[]
    for k in range(0,variables.L):
        if k==j:  
            var4.append(S_x())
        else:
            var4.append(identity(2))
    for i in range(0,variables.L-1):
        var4[i+1]=kron(var4[i],var4[i+1])
    return(var4[i+1])

def correlation_matrix_S_down_i_S_up_j():
    row=np.zeros(variables.L**2)
    col=np.zeros(variables.L**2)
    data=np.zeros(variables.L**2)
    kkkk=0
    for ii in range(0,variables.L):   
        for jj in range(0,variables.L):
            row[kkkk]=ii
            col[kkkk]=jj
            data[kkkk]=expectationsvalue_correlation_S_down_S_up(ii,jj).toarray()[0]
            kkkk+=1
    return csr_matrix((data,(row,col)),shape=(variables.L, variables.L))

def correlation_matrix_S_up_i_S_down_j():
    row=np.zeros(variables.L**2)
    col=np.zeros(variables.L**2)
    data=np.zeros(variables.L**2)
    kkkk=0
    for ii in range(0,variables.L):
        for jj in range(0,variables.L):
            row[kkkk]=ii
            col[kkkk]=jj
            data[kkkk]=expectationsvalue_correlation_S_up_S_down(ii,jj).toarray()[0] 
            kkkk+=1
    return csr_matrix((data,(row,col)),shape=(variables.L, variables.L))

def correlation_matrix_S_z_i_S_z_j():
    row=np.zeros(variables.L**2)
    col=np.zeros(variables.L**2)
    data=np.zeros(variables.L**2)
    kkkk=0
    for ii in range(0,variables.L):
        for jj in range(0,variables.L):
            row[kkkk]=ii
            col[kkkk]=jj
            data[kkkk]=expectationsvalue_correlation_Sz_Sz(ii,jj).toarray()[0]
            kkkk+=1
    return csr_matrix((data,(row,col)),shape=(variables.L, variables.L))

def expectationsvalue_correlation_Sz_Sz(i,j):
    var10=scipy.sparse.csr_matrix.dot(S_z_i(i),S_z_i(j))
    var11=scipy.sparse.csr_matrix.dot(var10,variables.groundstate)
    return scipy.sparse.csr_matrix.dot(var11.transpose(),variables.groundstate) 
                                  
def expectationsvalue_correlation_S_up_S_down(i,j):
    var12=S_down_up_i(i)
    var13=scipy.sparse.csr_matrix.dot(var12,variables.groundstate)
    return scipy.sparse.csr_matrix.dot(var13.transpose(),variables.groundstate) 
                            
def expectationsvalue_correlation_S_down_S_up(i,j):
    var12=S_down_up_i(i).transpose()
    var13=scipy.sparse.csr_matrix.dot(var12,variables.groundstate)
    return scipy.sparse.csr_matrix.dot(var13.transpose(),variables.groundstate) 

def expectationvalue_S_z_i(i):
    var5=scipy.sparse.csr_matrix.dot(S_z_i(i),variables.groundstate)
    return scipy.sparse.csr_matrix.dot(var5.transpose(),variables.groundstate)

def expectationsvalue_S_squared_i(i):
    var6=scipy.sparse.csr_matrix.dot(S_squared_i(i),variables.groundstate)
    return scipy.sparse.csr_matrix.dot(var6.transpose(),variables.groundstate)



def S_up_i_neu(i): 
    row=np.zeros((variables.lenbasisstates))
    col=np.zeros((variables.lenbasisstates))
    data=np.zeros((variables.lenbasisstates))
    for j in range(0,(variables.lenbasisstates)):
        state=variables.basis_index_element_map[j]
        k=m_s_state_i(j,i)
        if (k == variables.spinlist[i]):
                continue
        else:
            state_up = state[:]
            state_up[i] = state[i] + 1
            p=variables.basis_element_index_map[str(state_up)]
            data[j]= ladder_factor_up(j,i)
            row[j]=j
            col[j]=p
        
    return csr_matrix((data,(row,col)),shape=(variables.lenbasisstates, variables.lenbasisstates))

def S_down_i_neu(i): 
    row=np.zeros((variables.lenbasisstates))
    col=np.zeros((variables.lenbasisstates))
    data=np.zeros((variables.lenbasisstates))
    for j in range(0,(variables.lenbasisstates)):
        state=variables.basis_index_element_map[j]
        k=m_s_state_i(j,i)
        if (k == -(variables.spinlist[i])):
                continue
        else:
            state_up = state[:]
            state_up[i] = state[i] - 1
            p=variables.basis_element_index_map[str(state_up)]
            data[j]= ladder_factor_down(j,i)
            row[j]=j
            col[j]=p
        
    return csr_matrix((data,(row,col)),shape=(variables.lenbasisstates, variables.lenbasisstates))

def S_down_up(i):    
    row=np.zeros(variables.lenbasisstates)
    col=np.zeros(variables.lenbasisstates)
    data=np.zeros(variables.lenbasisstates)
    for j in range(0,variables.lenbasisstates):
        state=variables.basis_index_element_map[j]
        k=m_s_state_i(j,i%(variables.L))
        l=m_s_state_i(j,(i)%(variables.L))
        if (k == variables.spinlist[i%(variables.L)]):
                continue
        if (l == -variables.spinlist[(i+1)%(variables.L)]):
                continue
        else:
            data[j]= ladder_factor_down(j,(i)%(variables.L))*ladder_factor_up(j,i%(variables.L))
            state_up = state[:]
            state_up[i%(variables.L)] = state[i%(variables.L)] + 1
            state_up[(i)%(variables.L)] = state_up[(i)%(variables.L)] - 1
            p=variables.basis_element_index_map[str(state_up)]
            row[j]=j
            col[j]=p
    return csr_matrix((data,(row,col)),shape=(variables.lenbasisstates, variables.lenbasisstates))

def S_z_tot():
    var=0
    for i in range(0,variables.L):
        var+=S_z_i(i)
    return var

def S_x_tot():
    var=0
    for i in range(0,variables.L):
        var+=1/2*(S_up_i_neu(i)-S_down_i_neu(i))
    return var

def S_y_tot():
    var=0
    for i in range(0,variables.L):
        var+=1/2*(S_up_i_neu(i)+S_down_i_neu(i))
    return var

def S_x_i(i):
    return 1/2*(S_up_i_neu(i)+S_down_i_neu(i))

def S_y_i(i):
    return 1/4*scipy.sparse.csr_matrix.dot((S_up_i_neu(i)+S_down_i_neu(i)),((S_up_i_neu(i)+S_down_i_neu(i))))
import cmath

def S_y_i2(i):
    return -((1j)/2)*(S_up_i_neu(i)-S_down_i_neu(i))

def casimir_operator_i(i):
    return 1/4*scipy.sparse.csr_matrix.dot((S_up_i_neu(i)+S_down_i_neu(i)),((S_up_i_neu(i)+S_down_i_neu(i)))) - 1/4*scipy.sparse.csr_matrix.dot((S_up_i_neu(i)-S_down_i_neu(i)),((S_up_i_neu(i)-S_down_i_neu(i))))+scipy.sparse.csr_matrix.dot(S_z_i(i),S_z_i(i))

def casimir_operator_tot():
    return scipy.sparse.csr_matrix.dot(S_x_tot(),S_x_tot())+ scipy.sparse.csr_matrix.dot(S_y_tot(),S_y_tot())+ scipy.sparse.csr_matrix.dot(S_z_tot(),S_z_tot())+ scipy.sparse.csr_matrix.dot(S_z_tot(),S_z_tot())


def expectation(operator):
    return scipy.sparse.csr_matrix.dot(scipy.sparse.csr_matrix.dot(operator,variables.groundstate).transpose(),variables.groundstate).toarray()[0][0]

def commutator(A,B):
    c1=scipy.sparse.csr_matrix.dot(variables.groundstate.transpose(),scipy.sparse.csr_matrix.dot(B,scipy.sparse.csr_matrix.dot(A,variables.groundstate)))
    c2=scipy.sparse.csr_matrix.dot(variables.groundstate.transpose(),scipy.sparse.csr_matrix.dot(A,scipy.sparse.csr_matrix.dot(B,variables.groundstate)))
    return c1-c2

def back_to_full_basis(state):
    spinlist_total=variables.spinlist
    lenbasisstates_total=variables.lenbasisstates
    reducedbasis_index_element_map=variables.basis_index_element_map
    basis.createbasis(variables.spinlist)
    unreducedbasis_index_element_map=variables.basis_index_element_map
    unreducedbasis_element_index_map=variables.basis_element_index_map
    lenunreducedbasis_index_element_map=len(unreducedbasis_index_element_map)
    #psiarraydata=[i[0] for i in state.toarray()]
    psiarraydata=[variables.groundstate[i].toarray()[0][0] for i in range(0,max(variables.groundstate.shape))]
    row=[]
    for j in range(lenbasisstates_total):
        row.append(unreducedbasis_element_index_map[str(reducedbasis_index_element_map[j])])
    variables.groundstate=csr_matrix((psiarraydata,(row,np.zeros(lenbasisstates_total))),shape=(lenunreducedbasis_index_element_map, 1))
    #variables.states[0]=variables.groundstate.toarray()[0]