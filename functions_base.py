import numpy as np
import copy

############################## Power Flow & Injection ##############################

def get_PQ_inj(Ybus, V):
    """ Complex power injection """    
    # return -S to match the sign convention in pandapower 
    I = (Ybus @ V)
    S = np.multiply(V, np.conj(I))   
    return -S

def get_PQ_flow(net, Yf, Yt, V):
    """ Return complex power flows """
    # for transformers: HV = from, LV = to
    """ Get sorted lists of from and to branch ids """
    branch_id_from = np.concatenate((net.line.from_bus.values, net.trafo.hv_bus.values))
    branch_id_to = np.concatenate((net.line.to_bus.values, net.trafo.lv_bus.values))

    """ Get complex power flows at from and to ends of all branches from admittance matrices and voltage vector"""
    I_from = Yf @ V
    I_to = Yt @ V
    V_from = np.array(V[branch_id_from])
    V_to = np.array(V[branch_id_to])
    S_from = np.multiply(V_from, np.conj(I_from))
    S_to = np.multiply(V_to, np.conj(I_to))
    return S_from, S_to 

############################## Power Flow & Injection Partial Derivatives ##############################

def get_dS_dV(V, Vm, Ybus):
    """ Compute the partial derivatives of power injections and flows w.r.t. voltage magnitudes and angles """    
    I = Ybus @ V 
    dS_dVm = -np.diag(V) @ (np.diag(np.conj(I)) + np.conj(Ybus)@np.diag(np.conj(V)))@np.linalg.inv(np.diag(Vm))    
    dS_dtheta = -1j*np.diag(V) @ (np.diag(np.conj(I)) - np.conj(Ybus)@np.diag(np.conj(V)))
    return dS_dVm, dS_dtheta

def get_dS_dV_flow(Yf, Yt, V, Vm, net):
    """ Compute the partial derivatives of power flows w.r.t. voltage magnitudes and angles """
    If = Yf @ V
    It = Yt @ V
    E = np.linalg.inv(np.diag(Vm)) @ V
    from_bus = np.concatenate((net.line.from_bus.values, net.trafo.hv_bus.values))
    to_bus = np.concatenate((net.line.to_bus.values, net.trafo.lv_bus.values))
    nbranch = len(from_bus)

    Cf = np.zeros((len(from_bus), len(V))); Ct = np.zeros((len(to_bus), len(V)))    
    Cf[np.arange(nbranch), from_bus] = 1; Ct[np.arange(nbranch), to_bus] = 1

    dSf_dVm = np.diag(np.conj(If)) @ Cf @ np.diag(E) + np.diag(Cf @ V) @ np.conj(Yf) @ np.diag(np.conj(E))
    dSf_dtheta = 1j*np.diag(np.conj(If)) @ Cf @ np.diag(V) - 1j*np.diag(Cf @ V) @ np.conj(Yf) @ np.diag(np.conj(V))
    dSt_dVm = np.diag(np.conj(It)) @ Ct @ np.diag(E) + np.diag(Ct @ V) @ np.conj(Yt) @ np.diag(np.conj(E))
    dSt_dtheta = 1j*np.diag(np.conj(It)) @ Ct @ np.diag(V) - 1j*np.diag(Ct @ V) @ np.conj(Yt) @ np.diag(np.conj(V))

    return dSf_dVm, dSf_dtheta, dSt_dVm, dSt_dtheta

############################## Helper functions ##############################

def get_measurements_pu(net):
    """ Return measurement values and std deviations (p.u.) """
    n_meas = len(net.measurement)
    z = np.zeros(n_meas); stds = np.zeros(n_meas)    
    sn_mva = net.sn_mva
    
    # Iterate through measurements
    for i, meas_idx in enumerate(net.measurement.index):
        meas = net.measurement.loc[meas_idx]        
        if meas['measurement_type'] == 'v':
            # Voltage already in p.u.
            z[i] = meas['value']
            stds[i] = meas['std_dev']
        else:
            # Power in MW/Mvar, convert to p.u.
            z[i] = meas['value'] / sn_mva
            stds[i] = meas['std_dev'] / sn_mva    
    return z, stds   

def get_connected_elements(net, bus_idx_list): 
    """ Return elements directly connected to buses in bus_idx_list  """
    connected_lines = []; connected_trafos = []; connected_buses = []

    for bus_idx in bus_idx_list:     
        lines = net.line[(net.line.from_bus == bus_idx) | (net.line.to_bus == bus_idx)].index.tolist()
        trafos = net.trafo[(net.trafo.hv_bus == bus_idx) | (net.trafo.lv_bus == bus_idx)].index.tolist()
        connected_lines.extend(lines)
        connected_trafos.extend(trafos)
        # Find buses connected to the current bus
        for line_idx in lines:
            line = net.line.loc[line_idx]
            if line.from_bus == bus_idx:
                connected_buses.append(int(line.to_bus))
            else:
                connected_buses.append(int(line.from_bus))
        for trafo_idx in trafos:
            trafo = net.trafo.loc[trafo_idx]
            if trafo.hv_bus == bus_idx:
                connected_buses.append(int(trafo.lv_bus))
            else:
                connected_buses.append(int(trafo.hv_bus))

    return list(set(connected_lines)), list(set(connected_trafos)), list(set(connected_buses))

def get_measurements_elmt(net, elmt_id, elmt_type):
    """ Return measurement indices associated with elmt_id 
    elmt_type: 'bus', 'line' or 'trafo' """
    if elmt_type == "trafo" and elmt_id < 9:
        elmt_id = elmt_id + net.line.index.max() + 1
    
    meas_ids = net.measurement[(net.measurement.element_type == elmt_type) & (net.measurement.element == elmt_id)].index.tolist()
    meas_types = net.measurement.loc[meas_ids, "measurement_type"].tolist()

    return meas_ids, meas_types   
