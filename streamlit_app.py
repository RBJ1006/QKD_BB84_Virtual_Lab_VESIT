# streamlit_app.py

    
# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator

st.set_page_config(page_title="QXplore: QKD BB84 Virtual Lab", layout="wide")

# ---------- Helper Functions ----------
def generate_bb84_keys(n_qubits, eve_present=False, noise=0.0, distance=0.0):
    alice_bits = np.random.randint(2, size=n_qubits)
    alice_basis = np.random.choice(['Z', 'X'], size=n_qubits)
    bob_basis = np.random.choice(['Z', 'X'], size=n_qubits)
    eve_basis = np.random.choice(['Z', 'X'], size=n_qubits) if eve_present else None

    bob_results = []

    for i in range(n_qubits):
        bit = alice_bits[i]
        basis_match = False

        # Eve intervention
        if eve_present:
            # Eve measures the qubit
            if eve_basis[i] != alice_basis[i]:
                bit = np.random.randint(2)
        # Channel noise
        if np.random.rand() < noise:
            bit = 1 - bit
        bob_results.append(bit)

    bob_results = np.array(bob_results)

    # Basis match & status
    basis_match_arr = ['Yes' if alice_basis[i] == bob_basis[i] else 'No' for i in range(n_qubits)]
    status_arr = ['Kept' if bm == 'Yes' else 'Discarded' for bm in basis_match_arr]

    # Keys from kept bits
    key_indices = [i for i, s in enumerate(status_arr) if s == 'Kept']
    alice_key = ''.join([str(alice_bits[i]) for i in key_indices])
    bob_key = ''.join([str(bob_results[i]) for i in key_indices])

    # QBER
    if len(alice_key) > 0:
        errors = sum([alice_bits[i] != bob_results[i] for i in key_indices])
        qber = errors / len(alice_key)
    else:
        qber = 0.0

    # BB84 Table
    table_data = {
        'Alice Bit': alice_bits,
        'Alice Basis': alice_basis,
        'Bob Basis': bob_basis,
        'Bob Result': bob_results,
        'Basis Match': basis_match_arr,
        'Status': status_arr
    }
    if eve_present:
        table_data['Eve Basis'] = eve_basis
    bb84_df = pd.DataFrame(table_data)

    return bb84_df, alice_key, bob_key, qber

# ---------- Sidebar ----------
st.sidebar.title("QXplore: QKD Experiments")
experiment = st.sidebar.radio("Select Experiment", 
                              ["Experiment 1: BB84 with/without Eve",
                               "Experiment 2: Impact of Qubit Variation",
                               "Experiment 3: Impact of Channel Noise",
                               "Experiment 4: Impact of Distance"])

st.sidebar.markdown("---")

# ---------- Experiment 1 ----------
if experiment == "Experiment 1: BB84 with/without Eve":
    st.header("Experiment 1: Study of BB84 Protocol with and without Eve")
    st.subheader("Introduction / Aim")
    st.write("""
    In this experiment, we study the BB84 quantum key distribution protocol and observe the impact of an eavesdropper (Eve) on key generation.
    The aim is to understand how Eve’s presence affects the final key agreement between Alice and Bob and how it influences the Quantum Bit Error Rate (QBER).

    **Parameters:**
    - Number of Qubits (qubits): Total qubits transmitted from Alice to Bob.
    - Eve Presence (Yes/No): Whether Eve is intercepting the qubits.
    - QBER (%): Fraction of mismatched bits between Alice and Bob's keys.
    """)

    st.subheader("Controls / Procedure")
    n_qubits = st.number_input("Number of Qubits", min_value=8, max_value=128, value=16, step=8, help="Total number of qubits transmitted")
    eve_present = st.checkbox("Include Eve", value=False)
    run_exp = st.button("Run Experiment")
    reset_exp = st.button("Reset")

    if run_exp:
        bb84_df, alice_key, bob_key, qber = generate_bb84_keys(n_qubits, eve_present)
        st.subheader("BB84 Protocol Process")
        st.dataframe(bb84_df)
        st.write(f"**Alice Key:** {alice_key}")
        st.write(f"**Bob Key:** {bob_key}")
        st.write(f"**QBER:** {qber*100:.2f}%")

# ---------- Experiment 2 ----------
elif experiment == "Experiment 2: Impact of Qubit Variation":
    st.header("Experiment 2: Impact of Qubit Variation on BB84")
    st.subheader("Introduction / Aim")
    st.write("""
    This experiment investigates how varying the number of qubits affects the BB84 protocol.
    The aim is to analyze the relationship between the number of transmitted qubits and QBER, both in the presence and absence of Eve.

    **Parameters:**
    - Number of Qubits (qubits): 8 → 128, powers of 2.
    - Eve Presence (Yes/No)
    - QBER (%)
    """)

    qubit_list = st.multiselect("Select Number of Qubits (powers of 2)", options=[8,16,32,64,128], default=[8,16,32,64])
    eve_present = st.checkbox("Include Eve", value=False)
    run_exp = st.button("Run Experiment")
    reset_exp = st.button("Reset")

    if run_exp:
        summary_data = {'No. of Qubits': [], 'Key Generated': [], 'QBER': []}
        for n_qubits in qubit_list:
            bb84_df, alice_key, bob_key, qber = generate_bb84_keys(n_qubits, eve_present)
            st.subheader(f"BB84 Protocol Process for {n_qubits} qubits")
            st.dataframe(bb84_df)
            summary_data['No. of Qubits'].append(n_qubits)
            summary_data['Key Generated'].append(alice_key)
            summary_data['QBER'].append(qber*100)
        summary_df = pd.DataFrame(summary_data)
        st.subheader("Impact of Qubit Variation on QBER")
        st.dataframe(summary_df)
        fig, ax = plt.subplots()
        ax.plot(summary_df['No. of Qubits'], summary_df['QBER'], marker='o')
        ax.set_xlabel("Number of Qubits")
        ax.set_ylabel("QBER (%)")
        ax.set_title("QBER vs Number of Qubits")
        st.pyplot(fig)

# ---------- Experiment 3 ----------
elif experiment == "Experiment 3: Impact of Channel Noise":
    st.header("Experiment 3: Impact of Channel Noise on BB84")
    st.subheader("Introduction / Aim")
    st.write("""
    In this experiment, we study how channel noise affects the BB84 protocol.
    The aim is to understand how noise influences key agreement and QBER, both with and without an eavesdropper.

    **Parameters:**
    - Number of Qubits (qubits)
    - Channel Noise (probability): 0 → 0.2, e.g., 0.01 = 1% chance of bit flip.
    - Eve Presence (Yes/No)
    - QBER (%)
    """)

    n_qubits = st.number_input("Number of Qubits", min_value=8, max_value=128, value=16, step=8)
    noise_level = st.slider("Channel Noise (probability)", min_value=0.0, max_value=0.2, value=0.0, step=0.01)
    eve_present = st.checkbox("Include Eve", value=False)
    run_exp = st.button("Run Experiment")
    reset_exp = st.button("Reset")

    if run_exp:
        bb84_df, alice_key, bob_key, qber = generate_bb84_keys(n_qubits, eve_present, noise=noise_level)
        st.subheader(f"BB84 Protocol Process with Noise={noise_level*100:.2f}%")
        st.dataframe(bb84_df)
        st.write(f"**Alice Key:** {alice_key}")
        st.write(f"**Bob Key:** {bob_key}")
        st.write(f"**QBER:** {qber*100:.2f}%")

# ---------- Experiment 4 ----------
elif experiment == "Experiment 4: Impact of Distance":
    st.header("Experiment 4: Impact of Distance on BB84")
    st.subheader("Introduction / Aim")
    st.write("""
    This experiment studies how increasing the distance between Alice and Bob affects the BB84 protocol.
    The aim is to observe the effect of distance on key generation and QBER, both with and without an eavesdropper.

    **Parameters:**
    - Number of Qubits (qubits)
    - Distance (km)
    - Eve Presence (Yes/No)
    - QBER (%)
    """)

    n_qubits = st.number_input("Number of Qubits", min_value=8, max_value=128, value=16, step=8)
    distance = st.slider("Distance (km)", min_value=0, max_value=100, value=0, step=10)
    eve_present = st.checkbox("Include Eve", value=False)
    run_exp = st.button("Run Experiment")
    reset_exp = st.button("Reset")

    if run_exp:
        # We simulate distance effect as additional noise proportional to distance
        distance_noise = distance/500  # simple linear scaling
        bb84_df, alice_key, bob_key, qber = generate_bb84_keys(n_qubits, eve_present, noise=distance_noise)
        st.subheader(f"BB84 Protocol Process at Distance={distance} km")
        st.dataframe(bb84_df)
        st.write(f"**Alice Key:** {alice_key}")
        st.write(f"**Bob Key:** {bob_key}")
        st.write(f"**QBER:** {qber*100:.2f}%")

    

    
  
