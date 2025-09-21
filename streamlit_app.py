# Paste the full streamlit_app.py code here
# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile, assemble, execute
#from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

st.set_page_config(
    page_title="QXplore: Quantum Key Distribution (BB84 Protocol) Virtual Lab",
    layout="wide"
)

# ---------- Helper Functions ----------

def generate_bits(n):
    return np.random.randint(0, 2, n)

def generate_bases(n):
    return np.random.randint(0, 2, n)

def simulate_bb84(n_qubits=8, eve=False, noise=0.0, distance=0.0):
    """
    Simulate BB84 protocol.
    """
    # Alice generates bits and bases
    alice_bits = generate_bits(n_qubits)
    alice_bases = generate_bases(n_qubits)
    
    # Eve (optional)
    if eve:
        eve_bases = generate_bases(n_qubits)
        eve_bits = alice_bits.copy()  # Eve measures in her basis
        # Simulate Eve interception and possible bit flip
        flip_mask = eve_bases != alice_bases
        eve_bits[flip_mask] = np.random.randint(0,2, np.sum(flip_mask))
        transmitted_bits = eve_bits.copy()
        transmitted_bases = eve_bases.copy()
    else:
        transmitted_bits = alice_bits.copy()
        transmitted_bases = alice_bases.copy()
    
    # Bob measures
    bob_bases = generate_bases(n_qubits)
    bob_bits = transmitted_bits.copy()
    
    # Introduce noise
    flip_noise = np.random.rand(n_qubits) < noise
    bob_bits[flip_noise] = 1 - bob_bits[flip_noise]
    
    # Basis match & status
    basis_match = ["Yes" if alice_bases[i]==bob_bases[i] else "No" for i in range(n_qubits)]
    status = ["Kept" if m=="Yes" else "Discarded" for m in basis_match]
    
    # Keys
    alice_key = [alice_bits[i] for i in range(n_qubits) if status[i]=="Kept"]
    bob_key = [bob_bits[i] for i in range(n_qubits) if status[i]=="Kept"]
    
    # QBER
    qber = 0.0
    if len(alice_key)>0:
        qber = np.sum(np.array(alice_key) != np.array(bob_key)) / len(alice_key)
    
    # Build table
    table = pd.DataFrame({
        "Alice Bit": alice_bits,
        "Alice Basis": alice_bases,
        "Bob Basis": bob_bases,
        "Basis Match": basis_match,
        "Status": status,
        "Bob Bit": bob_bits
    })
    if eve:
        table["Eve Basis"] = eve_bases
        table["Eve Bit"] = eve_bits
    
    return table, alice_key, bob_key, qber

# ---------- Sidebar: Experiment Selection ----------
st.sidebar.title("Experiments")
experiment = st.sidebar.radio(
    "Select Experiment",
    (
        "Experiment 1: BB84 Protocol with/without Eve",
        "Experiment 2: Impact of Number of Qubits on QBER",
        "Experiment 3: Impact of Channel Noise on QBER",
        "Experiment 4: Impact of Distance on QBER"
    )
)

# ---------- Experiment 1 ----------
if experiment=="Experiment 1: BB84 Protocol with/without Eve":
    st.header("Experiment 1: BB84 Protocol with and without Eve")
    st.markdown("""
    **Aim:** Observe the impact of Eve on the BB84 protocol.  
    This experiment demonstrates how the presence or absence of an eavesdropper affects key agreement and QBER.
    """)
    
    # Controls
    n_qubits = st.sidebar.slider("Number of Qubits", 8, 128, 16, step=8)
    run_without_eve = st.sidebar.button("Run Without Eve")
    run_with_eve = st.sidebar.button("Run With Eve")
    compare_btn = st.sidebar.button("Compare QBER")
    
    # Run simulations
    if run_without_eve:
        table_no_eve, alice_key_no_eve, bob_key_no_eve, qber_no_eve = simulate_bb84(n_qubits=n_qubits, eve=False)
        st.subheader("BB84 Protocol Process (Without Eve)")
        st.dataframe(table_no_eve)
        st.write(f"Alice Key: {alice_key_no_eve}")
        st.write(f"Bob Key: {bob_key_no_eve}")
        st.write(f"QBER: {qber_no_eve:.2%}")
    
    if run_with_eve:
        table_eve, alice_key_eve, bob_key_eve, qber_eve = simulate_bb84(n_qubits=n_qubits, eve=True)
        st.subheader("BB84 Protocol Process (With Eve)")
        st.dataframe(table_eve)
        st.write(f"Alice Key: {alice_key_eve}")
        st.write(f"Bob Key: {bob_key_eve}")
        st.write(f"QBER: {qber_eve:.2%}")
    
    if compare_btn:
        # Both runs required
        if 'qber_no_eve' in locals() and 'qber_eve' in locals():
            fig, ax = plt.subplots()
            ax.bar(["Without Eve", "With Eve"], [qber_no_eve, qber_eve], color=["green","red"])
            ax.set_ylabel("QBER")
            ax.set_title("QBER Comparison")
            st.pyplot(fig)
        else:
            st.warning("Run both simulations first to compare QBER.")

# ---------- Experiment 2 ----------
elif experiment=="Experiment 2: Impact of Number of Qubits on QBER":
    st.header("Experiment 2: Impact of Number of Qubits on QBER")
    st.markdown("""
    **Aim:** Observe how increasing the number of qubits affects QBER in the BB84 protocol.  
    This experiment generates a key for each qubit count and computes QBER.
    """)
    
    qubit_list = [8, 16, 32, 64, 128]
    qber_without_eve_list = []
    qber_with_eve_list = []
    
    table_overall = pd.DataFrame(columns=["Number of Qubits", "Alice Key", "Bob Key", "QBER (No Eve)", "QBER (With Eve)"])
    
    for nq in qubit_list:
        t_no_eve, ak_no_eve, bk_no_eve, q_no_eve = simulate_bb84(n_qubits=nq, eve=False)
        t_eve, ak_eve, bk_eve, q_eve = simulate_bb84(n_qubits=nq, eve=True)
        table_overall.loc[len(table_overall)] = [nq, ak_no_eve, bk_no_eve, q_no_eve, q_eve]
        qber_without_eve_list.append(q_no_eve)
        qber_with_eve_list.append(q_eve)
    
    st.subheader("Impact of Number of Qubits on QBER")
    st.dataframe(table_overall)
    
    fig, ax = plt.subplots()
    ax.plot(qubit_list, qber_without_eve_list, marker='o', label="Without Eve")
    ax.plot(qubit_list, qber_with_eve_list, marker='s', label="With Eve")
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("QBER")
    ax.set_title("QBER vs Number of Qubits")
    ax.legend()
    st.pyplot(fig)

# ---------- Experiment 3 ----------
elif experiment=="Experiment 3: Impact of Channel Noise on QBER":
    st.header("Experiment 3: Impact of Channel Noise on QBER")
    st.markdown("""
    **Aim:** Study the effect of channel noise on QBER.  
    Noise represents the probability that a qubit flips during transmission.
    """)
    
    n_qubits = st.sidebar.slider("Number of Qubits", 8, 128, 32, step=8)
    noise_values = np.arange(0.0, 0.21, 0.05)
    qber_list_no_eve = []
    qber_list_eve = []
    
    for noise in noise_values:
        _, _, _, q_no_eve = simulate_bb84(n_qubits=n_qubits, eve=False, noise=noise)
        _, _, _, q_eve = simulate_bb84(n_qubits=n_qubits, eve=True, noise=noise)
        qber_list_no_eve.append(q_no_eve)
        qber_list_eve.append(q_eve)
    
    fig, ax = plt.subplots()
    ax.plot(noise_values, qber_list_no_eve, marker='o', label="Without Eve")
    ax.plot(noise_values, qber_list_eve, marker='s', label="With Eve")
    ax.set_xlabel("Channel Noise Probability")
    ax.set_ylabel("QBER")
    ax.set_title(f"QBER vs Channel Noise for {n_qubits} Qubits")
    ax.legend()
    st.pyplot(fig)

# ---------- Experiment 4 ----------
elif experiment=="Experiment 4: Impact of Distance on QBER":
    st.header("Experiment 4: Impact of Distance on QBER")
    st.markdown("""
    **Aim:** Study the effect of distance between Alice & Bob on QBER.  
    Distance is simulated as an effective noise probability increasing with separation.
    """)
    
    n_qubits = st.sidebar.slider("Number of Qubits", 8, 128, 32, step=8)
    distance_values = np.arange(0, 11, 2)  # e.g., 0 to 10 units
    qber_list_no_eve = []
    qber_list_eve = []
    
    for dist in distance_values:
        noise = dist * 0.01  # Example: 1% noise per unit distance
        _, _, _, q_no_eve = simulate_bb84(n_qubits=n_qubits, eve=False, noise=noise)
        _, _, _, q_eve = simulate_bb84(n_qubits=n_qubits, eve=True, noise=noise)
        qber_list_no_eve.append(q_no_eve)
        qber_list_eve.append(q_eve)
    
    fig, ax = plt.subplots()
    ax.plot(distance_values, qber_list_no_eve, marker='o', label="Without Eve")
    ax.plot(distance_values, qber_list_eve, marker='s', label="With Eve")
    ax.set_xlabel("Distance Units")
    ax.set_ylabel("QBER")
    ax.set_title(f"QBER vs Distance for {n_qubits} Qubits")
    ax.legend()
    st.pyplot(fig)


    
       
            
