import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

st.set_page_config(page_title="QXplore: BB84 Virtual Lab", layout="wide")

st.title("QXplore: Quantum Key Distribution (BB84 Protocol) Virtual Lab")

# Sidebar: Experiment selection
experiment = st.sidebar.selectbox("Select Experiment", 
                                  ["Experiment 1: BB84 with/without Eve",
                                   "Experiment 2: Impact of Qubits",
                                   "Experiment 3: Impact of Channel Noise",
                                   "Experiment 4: Impact of Distance"])

st.sidebar.write("Use the controls to configure and run the experiment.")

# Number of qubits control
num_qubits = st.sidebar.slider("Number of Qubits", min_value=8, max_value=128, step=2**1, value=8)

# Run buttons
run_without_eve = st.sidebar.button("Run Without Eve")
run_with_eve = st.sidebar.button("Run With Eve")
compare_qber = st.sidebar.button("Compare QBER")
reset_experiment = st.sidebar.button("Reset Experiment")

# Example: minimal BB84 simulation (without Eve)
if run_without_eve or run_with_eve:
    # Alice generates random bits and bases
    alice_bits = np.random.randint(2, size=num_qubits)
    alice_bases = np.random.randint(2, size=num_qubits)

    # Bob generates random bases
    bob_bases = np.random.randint(2, size=num_qubits)

    # Simulate photon transmission (BB84)
    simulator = AerSimulator()
    final_key = []
    rows = []

    for i in range(num_qubits):
        qc = QuantumCircuit(1, 1)
        # Encode Alice's bit in her basis
        if alice_bases[i] == 0:  # Z basis
            if alice_bits[i] == 1:
                qc.x(0)
        else:  # X basis
            if alice_bits[i] == 1:
                qc.x(0)
            qc.h(0)

        # Bob measures in his basis
        if bob_bases[i] == 1:
            qc.h(0)
        qc.measure(0, 0)

        job = simulator.run(transpile(qc, simulator), shots=1)
        result = job.result()
        counts = result.get_counts()
        bob_bit = int(list(counts.keys())[0])

        basis_match = "Yes" if alice_bases[i] == bob_bases[i] else "No"
        status = "Kept" if basis_match=="Yes" else "Discarded"
        if status=="Kept":
            final_key.append(bob_bit)

        rows.append([alice_bits[i], alice_bases[i], bob_bases[i], basis_match, status])

    df = pd.DataFrame(rows, columns=["Alice Bit", "Alice Basis", "Bob Basis", "Basis Match", "Status"])
    st.subheader("BB84 Protocol Process")
    st.dataframe(df)

    st.subheader("Final Key")
    st.text(final_key)

    qber = np.sum(alice_bits[:len(final_key)] != final_key) / len(final_key) if len(final_key)>0 else 0
    st.write(f"Quantum Bit Error Rate (QBER): {qber:.2f}")

  

    
    
   
    
    
   
   
