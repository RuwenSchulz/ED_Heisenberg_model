import scipy
from scipy.sparse import diags, kron, csr_matrix
from scipy.sparse.linalg import eigs, eigsh


import ed_1D_spinchain.variables as variables
from ed_1D_spinchain.basis import filter_basisstates
from ed_1D_spinchain.operators import S_z_i,S_x_i,S_down_up_i,getstate


def Hamiltonian(bc=0,D=0,J_z=1,J_xy=1,h=0,g=0,k=3,state=0,block=False,pin="off"): 
    states_and_spectrum = []    
    if(type(block)==int):
        filter_basisstates(block)
    H_sparse=csr_matrix((variables.lenbasisstates,variables.lenbasisstates))
    if(pin!="off"):
        for i in pin:
            H_sparse -= i[1]*S_z_i(i[0])  
    for i in range(0,variables.L-1):
        if(g!=0):
            H_sparse -= g*S_x_i(i)     
        if(h!=0):
            H_sparse -= h*S_z_i(i)
        H_sparse += D*csr_matrix.dot(S_z_i(i),S_z_i(i))
        H_sparse += J_z*csr_matrix.dot(S_z_i(i),S_z_i(i+1))
        
        H_sparse += 1/2*J_xy*S_down_up_i(i)
        H_sparse += 1/2*J_xy*S_down_up_i(i).transpose()
    if(g!=0):
        H_sparse -= g*S_x_i(variables.L-1)
    if(h!=0):
        H_sparse -= h*S_z_i(variables.L-1)
    H_sparse += D*csr_matrix.dot(S_z_i(variables.L-1),S_z_i(variables.L-1))
    
    #BC:
    if(bc!=0):
        phase_factor=bc
        H_sparse += J_z*phase_factor*csr_matrix.dot(S_z_i(variables.L-1),S_z_i(0))
        H_sparse += 1/2*J_xy*phase_factor*S_down_up_i(variables.L-1)
        H_sparse += 1/2*J_xy*phase_factor*S_down_up_i(variables.L-1).transpose()
    states_and_spectrum = eigsh(H_sparse,k=k,which='SA')

    variables.states=states_and_spectrum[1]
    variables.spectrum=states_and_spectrum[0]
    variables.groundstate=getstate(state)
    
    #print("created hamiltonian:",bc,D,J_z,J_xy,h,g,k,state,block,end='\r')
    
    return H_sparse
