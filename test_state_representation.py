from time import time
import numpy as np
from numpy.linalg import matrix_power
from numpy.random import default_rng
from scipy.sparse import csr_matrix
from scipy.special import binom
from math import factorial

# simulation parameters
Truncation = 2 # global truncation
Formalism = "dm" # global formalism of photonic state
SPDC1_mean = 0.1 # mean photon number in one output mode of SPDC source 1
SPDC2_mean = 0.1 # mean photon number in one output mode of SPDC source 2
Memo1_abs_effi = 0.9 # memory 1 absorption efficiency
Memo2_abs_effi = 0.9 # memory 2 absorption efficiency
Bsm1_loss = 0.9 # loss between memory 1 and BSM
Bsm2_loss = 0.9 # loss between memory 2 and BSM
Bsm1_effi = 0.9 # efficiency of photon detector 1 for BSM
Bsm2_effi = 0.9 # efficiency of photon detector 2 for BSM
DM1_effi = 0.9 # efficiency of photon detector 1 for density matrix measurement
DM2_effi = 0.9 # efficiency of photon detector 2 for density matrix measurement
Phase_bsm = 0 # relative phase between two optical paths from source to BSM
Pulse_num = 1000 # number of pulses to generate

# TODO: decay as function of time (in final implementation)
Memo1_effi_decay_rate = 0. # decay rate of memory 1 retrieval efficiency, modeled as exponential decay
Memo2_effi_decay_rate = 0. # decay rate of memory 2 retrieval efficiency, modeled as exponential decay


def build_basis(truncation, number):
    """Generate basis vector of truncated Fock space with a certain excitation number."""
    basis = np.zeros(truncation+1)
    basis[number] = 1
    
    return basis


def build_create(truncation):
    """Generate matrix of creation operator on truncated Hilbert space."""
    data = np.array([np.sqrt(i+1) for i in range(truncation)]) # elements in create/annihilation operator matrix
    row = np.array([i+1 for i in range(truncation)])
    col = np.array([i for i in range(truncation)])
    create = csr_matrix((data, (row, col)), size=(truncation+1, truncation+1)).toarray()

    return create


def build_povm1(truncation, create, destroy):
    """Generate matrix of POVM operator representing a photon detector having 1 click."""
    series_elem_list = [(-1)**i*matrix_power(create,i+1).dot(matrix_power(destroy,i+1))/factorial(i+1) for i in range(truncation)]
    povm1 = sum(series_elem_list)

    return povm1


def build_kraus_ops(truncation, loss_rate):
    """Generate list of Kraus operators for the truncated generalized amplitude damping channel for one mode."""
    kraus_ops = []
    for k in range(truncation+1):
        kraus_op = np.zeros((truncation+1,truncation+1))
        for n in range(truncation+1-k):
            coeff = np.sqrt(binom(n+k,k))*np.sqrt((1-loss_rate)**(n)*loss_rate**k)
            op = np.dot(build_basis(truncation,n), build_basis(truncation,n+k).H) # transition operator
            kraus_op += coeff*op
        kraus_ops.append(kraus_op)

    return kraus_ops


def apply_quantum_channel(state, kraus_ops):
    """Apply quantum channel on quantum state.
    
    Arguments:
        state (array): Quantum state *density matrix* as numpy array.
        kraus_ops (list[array]): List of Kraus operators' matrices as numpy array with dimensions compatible with quantum state.
    """
    shape = state.shape
    if len(shape) != 2:
        raise ValueError("Input state needs to be density matrix. Invalid input state " + state)
    elif shape[0] != shape[1]:
        raise ValueError("Input state needs to be density matrix. Invalid input state " + state)
    for op in kraus_ops:
        if op.shape != shape:
            raise ValueError("Kraus operators need to have compatible dimension with state. Invalid Kraus operator list " + kraus_ops)

    new_state = np.zeros(shape)
    for op in kraus_ops:
        new_state += op.dot(state).dot(op.H)

    return new_state


def measure_dm(state, meas_op):
    """Determine post measurement state in *density matrix* formalism given measurement operator.
        
    Arguments:
        state (array): Quantum state *density matrix* as numpy array.
        meas_op (array): Measurement operator with dimension compatible with quantum state.
    """
    shape = state.shape
    if len(shape) != 2:
        raise ValueError("Input state needs to be density matrix. Invalid state shape " + state)
    elif shape[0] != shape[1]:
        raise ValueError("Input state needs to be density matrix. Invalid state shape " + state)
    if meas_op.shape != shape:
        raise ValueError("Measurement operator needs to have compatible dimension with state. Invalid Kraus operator list " + meas_op)

    post_meas_dm = meas_op.dot(state).dot(meas_op.H)/np.trace(meas_op.dot(state).dot(meas_op.H))
    
    return post_meas_dm


def build_bell_state(truncation, sign, phase=0, formalism="dm"):
    """Generate standard Bell state which is heralded in ideal BSM for comparison with results from imperfect parameter choices."""
    basis0 = build_basis(truncation, 0)
    basis1 = build_basis(truncation, 1)
    basis10 = np.kron(basis1, basis0)
    basis01 = np.kron(basis0, basis1)
    
    if sign == "plus":
        ket = (basis10 + np.exp(1j*phase)*basis01)/np.sqrt(2)
    elif sign == "minus":
        ket = (basis10 - np.exp(1j*phase)*basis01)/np.sqrt(2)
    else:
        raise ValueError("Invalid Bell state sign type " + sign)

    dm = np.dot(ket, ket.H)

    if formalism == "dm":
        return dm
    elif formalism == "ket":
        return ket
    else:
        raise ValueError("Invalid quantum state formalism " + formalism)


def partial_trace_bsm(state, truncation):
    """Trace out the two measured photonic subsystems and return a composite state stored in two quantum memories."""
    shape = state.shape
    if len(shape) != 2:
        raise ValueError("Input state needs to be density matrix. Invalid state shape " + state)
    elif shape[0] != shape[1]:
        raise ValueError("Input state needs to be density matrix. Invalid state shape " + state)
    elif shape[0] != (truncation+1)**4:
        raise ValueError("Input state needs to be constructed by four subsystems of same dimension. Invalid state dimension " + state)

    state_tr_bsm1 = np.trace(state.reshape((truncation+1,)*8), axis1=1, axis2=5) # trace out subsystem measured by bsm1
    state_trtr_bsm2 = np.trace(state_tr_bsm1.reshape((truncation+1,)*6), axis1=1, axis2=4) # further trace out subsystem measured by bsm2

    state_trtr_bsm2.reshape(((truncation+1)**2,)*2)

    return state_trtr_bsm2


class SPDCOutputState():
    """
    SPDC source modeled as two-mode squeezed vacuum (TMSV) state generator.
    In real implementation the two subsystems need to be labeled
    so that separate operations on subsystems can be implemented acurately.

    Attributes :
        mean_num (float): Mean number of photon in one output mode. The default is 0.1.
        truncation (int): Maximal *excitation number* in one output mode. The default is 2.
        formalism (str): Formalism of output state. "dm" for density matrix and "ket" for state vector. The default is "dm".
        state (array): Quantum state of the output from SPDC source.
    """
    def __init__(self, mean_num=0.1, truncation=2, formalism="dm") -> None:
        self.mean_num = mean_num
        self.truncation = truncation
        self.formalism = formalism

        # create state component amplitudes list
        amp_list = [(np.sqrt(mean_num/(mean_num+1)))**m/np.sqrt(mean_num+1) for m in range(truncation)]
        amp_square_list = [amp**2 for amp in amp_list]
        amp_list.append(np.sqrt(1-sum(amp_square_list)))
        
        # create two-mode state vector
        state_vec = np.zeros((self.truncation+1)**2)

        for i in range(self.truncation+1):
            amp = amp_list[i]
            basis = build_basis(self.truncation,i)
            basis = np.kron(basis,basis)
            state_vec += amp*basis

        # create density matrix 
        state_dm = np.dot(state_vec,state_vec.H)

        if self.formalism == "dm":
            self.state = state_dm
        elif self.formalism == "ket":
            self.state = state_vec
        else:
            raise ValueError("Invalid quantum state formalism " + self.formalism)


class PhotonDetector():
    """
    Non-number-resolving photon detectors. 
    Modeled by two POVM operators and their corresponding measurement operators.

    Attributes :
        truncation (int): Maximal *excitation number* in one output mode. Should be determined by input state.
        efficiency (float): Detector efficiency between 0 and 1. The default is 1.
        seed (int): Seed for random number generator which is used to determine POVM outcome from the probability distribution. The default is 0.
        povm1 (array): The matrix of POVM operator corresponding to having 1 click.
        povm0 (array): The matrix of POVM operator corresponding to having 0 click.
        meas1 (array): The matrix of measurement operator determined by square root of povm1.
        meas0 (array): The matrix of measurement operator determined by square root of povm0.
        rng: Random number generator.
    """
    def __init__(self, truncation, efficiency=1, seed=0) -> None:
        self.truncation = truncation
        self.efficiency = efficiency
        self.seed = seed
        self.rng = default_rng(self.seed)

        # generate creation and annihilation operators for one mode
        identity = np.eye(self.truncation+1)
        create = build_create(self.truncation)
        destroy = create.H

        # generate POVM operators and measurement for one mode
        self.povm1 = build_povm1(self.truncation, self.efficiency*create, self.efficiency*destroy)
        self.povm0 = identity - self.povm1
        self.meas1 = matrix_power(self.povm1,1/2)
        self.meas0 = matrix_power(self.povm0,1/2)


class BeamsplitterMeas():
    """
    Measurement device with two photon detectors behind a beamsplitter.

    Attributes:
        truncation (int): Maximal *excitation number* in one output mode. Should be determined by input state.
        efficiency1 (float): Detector 1 efficiency between 0 and 1. The default is 1.
        efficiency2 (float): Detector 2 efficiency between 0 and 1. The default is 1.
        seed (int): Seed for random number generator which is used to determine POVM outcome from the probability distribution. The default is 0.
        phase (float): Relative phase between two optical paths to BSM node. The default is 0.
        povm1 (array): The matrix of POVM operator corresponding to PD-1 1 click.
        povm2 (array): The matrix of POVM operator corresponding to PD-2 1 click.
    """
    def __init__(self, truncation, efficiency1=1, efficiency2=1, phase=0, seed=0) -> None:
        self.efficiency1 = efficiency1
        self.efficiency2 = efficiency2
        self.truncation = truncation
        self.seed = seed
        self.rng = default_rng(self.seed)
        self.phase = phase

        identity = np.eye(self.truncation+1)
        create = build_create(self.truncation)

        # Modified mode operators in Heisenberg picture by beamsplitter transformation considering inefficiency (ignoring relative phase)
        create1 = (np.kron(self.efficiency1*create,identity) + np.exp(1j*self.phase)*np.kron(identity,self.efficiency2*create))/np.sqrt(2)
        destroy1 = create1.H
        create2 = (np.kron(self.efficiency1*create,identity) - np.exp(1j*self.phase)*np.kron(identity,self.efficiency2*create))/np.sqrt(2)
        destroy2 = create2.H

        self.povm1 = build_povm1(self.truncation,create1,destroy1)
        self.povm2 = build_povm1(self.truncation,create2,destroy2)


class BSM(BeamsplitterMeas):
    """
    Bell state measurement device with two photon detectors behind a beamsplitter.
    Modeled by two POVM operators and their corresponding measurement operators.
    
    Attributes :
        truncation (int): Maximal *excitation number* in one output mode. Should be determined by input state.
        efficiency1 (float): Detector 1 efficiency between 0 and 1. The default is 1.
        efficiency2 (float): Detector 2 efficiency between 0 and 1. The default is 1.
        seed (int): Seed for random number generator which is used to determine POVM outcome from the probability distribution. The default is 0.
        phase (float): Relative phase between two optical paths to BSM node. The default is 0.
        bsm1 (array): The matrix of POVM operator corresponding to PD-1 0 click, PD-2 1 click.
        bsm2 (array): The matrix of POVM operator corresponding to PD-1 1 click, PD-2 0 click.
        meas1 (array): The matrix of measurement operator determined by square root of povm10.
        meas2 (array): The matrix of measurement operator determined by square root of povm01.
        rng: Random number generator.
    """
    def __init__(self):
        super().__init__(truncation, efficiency1=1, efficiency2=1, phase=0, seed=0)
        
        identity = np.eye(self.truncation+1)
        self.bsm1 = np.dot(self.povm1,np.kron(identity,identity)-self.povm2)
        self.bsm2 = np.dot(np.kron(identity,identity)-self.povm1,self.povm2)

        self.meas1 = matrix_power(self.bsm1,1/2)
        self.meas2 = matrix_power(self.bsm2,1/2)


class DMDiagonalMeas():
    """
    Effective density matrix off-diagonal elements measurement device.
    Modeled by two POVM operators.

    Attributes : 
        truncation (int): Maximal *excitation number* in one output mode. Should be determined by input state.
        efficiency1 (float): Detector 1 efficiency between 0 and 1. The default is 1.
        efficiency2 (float): Detector 2 efficiency between 0 and 1. The default is 1.
        seed (int): Seed for random number generator which is used to determine POVM outcome from the probability distribution. The default is 0.
        povm00 (array): The matrix of POVM operator corresponding to PD-1 0 click, PD-2 0 click.
        povm01 (array): The matrix of POVM operator corresponding to PD-1 0 click, PD-2 1 click.
        povm10 (array): The matrix of POVM operator corresponding to PD-1 1 click, PD-2 0 click.
        povm11 (array): The matrix of POVM operator corresponding to PD-1 1 click, PD-2 1 click.
        rng: Random number generator.
    """
    def __init__(self, truncation, efficiency1=1, efficiency2=1, seed=0) -> None:
        self.truncation = truncation
        self.efficiency1 = efficiency1
        self.efficiency2 = efficiency2
        self.seed = seed
        self.rng = default_rng(self.seed)

        pd1 = PhotonDetector(self.truncation, self.efficiency1)
        pd2 = PhotonDetector(self.truncation, self.efficiency2)

        self.povm00 = np.kron(pd1.povm0,pd2.povm0)
        self.povm01 = np.kron(pd1.povm0,pd2.povm1)
        self.povm10 = np.kron(pd1.povm1,pd2.povm0)
        self.povm11 = np.kron(pd1.povm1,pd2.povm1)


class DMOffDiagonalMeas(BeamsplitterMeas):
    """
    Effective density matrix off-diagonal elements measurement device.
    Modeled by two POVM operators.

    Attributes : 
        truncation (int): Maximal *excitation number* in one output mode. Should be determined by input state.
        efficiency1 (float): Detector 1 efficiency between 0 and 1. The default is 1.
        efficiency2 (float): Detector 2 efficiency between 0 and 1. The default is 1.
        seed (int): Seed for random number generator which is used to determine POVM outcome from the probability distribution. The default is 0.
        phase (float): Relative phase between two modes to obtain interference. Unit in rad.
        povm1 (array): The matrix of POVM operator corresponding to PD 1 behind BS having 1 click.
        povm2 (array): The matrix of POVM operator corresponding to PD 2 behind BS having 1 click.
        rng: Random number generator.
    """
    def __init__(self):
        super().__init__(truncation, efficiency1=1, efficiency2=1, phase=0, seed=0)


def run_simulation(pulse_number, truncation, formalism, bsm1_effi, bsm2_effi, bsm1_loss, bsm2_loss, \
    memo1_abs_effi, memo2_abs_effi, spdc1_mean, spdc2_mean, phase_bsm):
    """
    Main function of simulation.
    Only for verification of concept, and for simplicity density matrix measurement part is not covered.

    Arguments:
        pulse_number (int): Number of pulses to generate in simulation.
        truncation (int): Global truncation of single-mode Fock space for all involved quantum (photonic) states.
        formalism (str): Global quantum state representation formalism for all involved quantum (photonic) states.
        bsm1_effi (float): Efficiency of PD1 of BSM.
        bsm2_effi (float): Efficiency of PD2 of BSM.
        bsm1_loss (float): Photon loss during transmission from source 1 to BSM (of optical fibre).
        bsm2_loss (float): Photon loss during transmission from source 2 to BSM (of optical fibre).
        memo1_abs_effi (float): Absorption efficiency of memory 1.
        memo2_abs_effi (float): Absorption efficiency of memory 2.
        spdc1_mean (float): Mean photon number of SPDC source 1 in one output mode.
        spdc2_mean (float): Mean photon number of SPDC source 2 in one output mode.
        phase_bsm (float): Relative phase between two optical paths from source to BSM.
    """
    # SPDC source output one by one, currently only for verification of concept so memory multimodality is not considered
    heralded_states = []
    fidelity_list = []

    for i in range(pulse_number):
        # initailization
        bsm_seed = i
        initial_state1 = SPDCOutputState(mean_num=spdc1_mean, truncation=Truncation, formalism=Formalism)
        initial_state2 = SPDCOutputState(mean_num=spdc2_mean, truncation=Truncation, formalism=Formalism)
        bsm = BSM(truncation, efficiency1=bsm1_effi, efficiency2=bsm2_effi, seed=bsm_seed)
        
        if formalism == "dm":
            joint_dm_init = np.kron(initial_state1.state, initial_state2.state)
        elif formalism == "ket":
            joint_dm_init = np.kron(initial_state1.state.dot(initial_state1.state.H), initial_state2.state.dot(initial_state2.state.H))
        else:
            raise ValueError("Invalid quantum state formalism " + formalism)

        # apply photon loss channels to state, each channel applies to only a subsystem so they should commute, order of application should not matter
        # 1st memo1 abs loss
        memo1_abs_kraus_ops = build_kraus_ops(truncation, 1-memo1_abs_effi)
        for i in len(memo1_abs_kraus_ops):
            memo1_abs_kraus_ops[i] = np.kron(np.kron(memo1_abs_kraus_ops[i],np.eye(truncation+1)),np.eye((truncation+1)**2))
        state_memo1_loss = apply_quantum_channel(joint_dm_init, memo1_abs_kraus_ops)
        # 2nd memo2 abs loss
        memo2_abs_kraus_ops = build_kraus_ops(truncation, 1-memo2_abs_effi)
        for i in len(memo2_abs_kraus_ops):
            memo2_abs_kraus_ops[i] = np.kron(np.eye((truncation+1)**2),np.kron(np.eye(truncation+1),memo2_abs_kraus_ops[i]))
        state_memo2_loss = apply_quantum_channel(state_memo1_loss, memo2_abs_kraus_ops)
        # 3rd bsm1 transmission loss
        bsm1_loss_kraus_ops = build_kraus_ops(truncation, bsm1_loss)
        for i in len(bsm1_loss_kraus_ops):
            bsm1_loss_kraus_ops[i] = np.kron(np.kron(np.eye(truncation+1),bsm1_loss_kraus_ops[i]),np.eye((truncation+1)**2))
        state_bsm1_loss = apply_quantum_channel(state_memo2_loss, bsm1_loss_kraus_ops)
        # 4th bsm2 transmission loss
        bsm2_loss_kraus_ops = build_kraus_ops(truncation, bsm2_loss)
        for i in len(bsm2_loss_kraus_ops):
            bsm2_loss_kraus_ops[i] = np.kron(np.eye((truncation+1)**2),np.kron(bsm2_loss_kraus_ops[i],np.eye(truncation+1)))
        joint_dm_pre_bsm = apply_quantum_channel(state_memo2_loss, bsm1_loss_kraus_ops)

        # determine BSM outcome
        bsm1_tot = np.eye(truncation+1).dot(bsm.bsm1).dot(np.eye(truncation+1))
        meas1_tot = matrix_power(bsm1_tot, 1/2)
        bsm2_tot = np.eye(truncation+1).dot(bsm.bsm2).dot(np.eye(truncation+1))
        meas2_tot = matrix_power(bsm2_tot, 1/2)
        prob1 = np.trace(joint_dm_pre_bsm.dot(bsm1_tot))
        prob2 = np.trace(joint_dm_pre_bsm.dot(bsm2_tot))
        prob0 = 1 - prob1 - prob2
        
        outcome = bsm.rng.choice(np.arange(3), p=[prob0, prob1, prob2])
        if outcome == 0:
            continue
        elif outcome == 1:
            sign = "plus"
            # post-measurement state
            joint_dm_post_bsm = measure_dm(joint_dm_pre_bsm, meas1_tot)
        else:
            sign = "minus"
            # post-measurement state
            joint_dm_post_bsm = measure_dm(joint_dm_pre_bsm, meas2_tot)

        # state stored in two memories conditional on 
        memo_state = partial_trace_bsm(joint_dm_post_bsm, truncation) 
        reference_bell_state = build_bell_state(truncation, sign, phase=phase_bsm)
        heralded_states.append({"state": memo_state, "sign": sign, "phase": phase_bsm})
        fidelity = np.trace(memo_state.dot(reference_bell_state)) # fidelity of stored state with respect to pure standard state
        fidelity_list.append(fidelity)
    
    heralded_num = len(fidelity_list)
    heralded_prob = heralded_num/pulse_number
    fidelity_avg = np.mean(fidelity_list)

    return heralded_states, fidelity_list, heralded_num, heralded_prob, fidelity_avg


if __name__ == "__main__":

    tick = time()
    results = run_simulation(Pulse_num, Truncation, Formalism, Bsm1_effi, Bsm2_effi, Bsm1_loss, Bsm2_loss, \
        Memo1_abs_effi, Memo2_abs_effi, SPDC1_mean, SPDC2_mean, Phase_bsm)
    sim_time = time() - tick

    print("Total simulation time: ", sim_time)
    print("Average time per pulse: ", sim_time / Pulse_num)
    print("-"*30)
    print("The heralding probability is " + results[3])
    print("The average fidelity of heralded entangled states is " + results[4])