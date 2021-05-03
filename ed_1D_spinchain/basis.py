#import needed functions
import numpy as np
from numpy.lib.scimath import logn
from scipy.sparse import diags, kron, csr_matrix
from scipy.sparse.linalg import eigs, eigsh

#import local functions
import variables as variables
from operators import m_s_state_i


def createbasis(inputchain):
    variables.basis_element_index_map={}
    variables.basis_index_element_map={}
    variables.spinlist=inputchain
    variables.L=len(inputchain)
    S=max(inputchain)
    basisn=int(2*S+1)
    basisstates=np.arange(int((2*S+1)**variables.L))
    kk=0
    for j in range(int((2*S+1)**variables.L)): 
        x= np.base_repr(j,basisn,variables.L if j == 0 else variables.L-1-int(logn(basisn,j)+0.000000000000001))
        y=[]
        for i in range(0,len(x)):   
            y.append(int(x[i]))  
        if(all((y[i]<=2*variables.spinlist[i] for i in range(0,len(y))))):    
            variables.basis_element_index_map[str(y)]=kk
            variables.basis_index_element_map[kk]=y
            kk+=1
        else:
            continue   
    variables.lenbasisstates=kk
    print("created basis!")
    
    
    
def createbasis_fast(inputchain):
    L=len(inputchain)
    variables.spinlist=inputchain
    def ternary3(n):
        xxx=np.base_repr(n,base=3)
        return [int(i) for i in (L-len(xxx))*'0'+xxx]
    def ternary3str(n):
        xxx=np.base_repr(n,base=3)
        return str([int(i) for i in (L-len(xxx))*'0'+xxx])
    S=max(inputchain)  
    L=len(inputchain)
    variables.basis_index_element_map=dict((h, ternary3(h)) for h in np.arange(int((2*S+1)**L)))
    variables.basis_element_index_map=dict((ternary3str(h), h) for h in np.arange(int((2*S+1)**L)))
    variables.lenbasisstates=(2*S+1)**L
    variables.L=L
    
def createbasis_block(two_m_tot,inputchain):
    filtered_basisstatesmap = {}
    mapfiltered_basisstates = {}
   
    #state to number
    variables.basis_element_index_map={}
    #number to state
    variables.basis_index_element_map={}
    variables.spinlist=inputchain
    variables.L=len(inputchain)
    S=max(inputchain)
    basisn=int(2*S+1)
    basisstates=np.arange(int((2*S+1)**variables.L))
    kk=0
    for j in range(int((2*S+1)**variables.L)): 
        x= np.base_repr(j,basisn,variables.L if j == 0 else variables.L-1-int(logn(basisn,j)+0.000000000000001))
        y=[]
        for i in range(0,len(x)):   
            y.append(int(x[i]))  
        if(all((y[i]<=2*variables.spinlist[i] for i in range(0,len(y))))):
            m_tot_i=0
            for j in range(0,variables.L):
                m_tot_i+=int(y[j])-(variables.spinlist[j])
            
            if (2*m_tot_i==two_m_tot):
                variables.basis_element_index_map[str(y)]=kk
                variables.basis_index_element_map[kk]=y
                kk+=1
            m_tot_i=0
        else:
            continue   
    variables.lenbasisstates=kk
    
def createbasis_block_fast(inputchain,block):
    L=len(inputchain)
    variables.spinlist=inputchain
    def ternary3(n):
        xxx=np.base_repr(n,base=3)
        return [int(i) for i in (L-len(xxx))*'0'+xxx]
    def ternary3str(n):
        xxx=np.base_repr(n,base=3)
        return str([int(i) for i in (L-len(xxx))*'0'+xxx])
    S=max(inputchain)  
    L=len(inputchain)
    variables.basis_index_element_map=dict((h, ternary3(h)) for h in np.arange(int((2*S+1)**L)) if((sum([int(ternary3(h)[j])-(variables.spinlist[j]) for j in range(variables.L)])==block) ))
    variables.lenbasisstates=(2*S+1)**L
    variables.L=L


    
def filter_basisstates(two_m_tot):
    filtered_basisstatesmap = {}
    mapfiltered_basisstates = {}
   
    m_tot_i=0
    kkkkk=0
    for i in range(0,variables.lenbasisstates):
        for j in range(0,variables.L):
            m_tot_i+=m_s_state_i(i,j)
        if (2*m_tot_i==two_m_tot):
            filtered_basisstatesmap[kkkkk]= variables.basis_index_element_map[i]
            mapfiltered_basisstates[str(variables.basis_index_element_map[i])]=kkkkk
            kkkkk+=1
        m_tot_i=0
    variables.basis_element_index_map=mapfiltered_basisstates
    variables.basis_index_element_map=filtered_basisstatesmap
    variables.lenbasisstates=kkkkk

def m_s_state_i(i,j):
    var6=variables.basis_index_element_map[i]
    m_s_i_j=int(var6[j])-(variables.spinlist[j])
    return m_s_i_j

def sort_basis_s_z():

    S=variables.L*max(variables.spinlist)
    multiplicity_m_z=(2*S)+1
    sorted_basis_index_to_state={}
    sorted_basis_state_to_index={}
    two_m_tot=int(-S*2)
    kkkkk=0
    #print(two_m_tot)
    #print(multiplicity_m_z)
    for j in range(multiplicity_m_z):
        #print("two_m_tot",two_m_tot)
        m_tot_i=0
        
        for i in range(0,variables.lenbasisstates):
            for j in range(0,variables.L):
                m_tot_i+=m_s_state_i(i,j)
            
            #print(m_tot_i)
            if (2*m_tot_i==two_m_tot):
                #print("hi")
                sorted_basis_index_to_state[kkkkk]= variables.basis_index_element_map[i]
                sorted_basis_state_to_index[str(variables.basis_index_element_map[i])]=kkkkk
                kkkkk+=1
            m_tot_i=0 
            
        two_m_tot+=2
    variables.basis_element_index_map=sorted_basis_state_to_index
    variables.basis_index_element_map=sorted_basis_index_to_state
    variables.lenbasisstates=kkkkk
