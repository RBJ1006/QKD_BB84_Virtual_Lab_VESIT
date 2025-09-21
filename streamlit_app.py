# Paste the full streamlit_app.py code here
# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

st.set_page_config(
    page_title="QXplore: Quantum Key Distribution (BB84 Protocol) Virtual Lab",
    layout="wide"
)

st.title("QXplore: Quantum Key Distribution (BB84 Protocol) Virtual Lab")

# Sidebar: Experiments
experiment = st.sidebar.selectbox("Select Experiment", 
                                  ["Experiment 1: BB84 with/without Eve",
                                   "Experiment 2: Qubit Variation Impact on QBER",
                                   "Experiment 3: Channel Noise Impact on QBER",
                                   "Experiment 4: Distance Impact on QBER"])

# Sidebar controls
num_qubits = st.sidebar.slider("Number of Qubits", min_value=8, max_value=128, step=8, value=16)
run_with_eve = st.sidebar.checkbox("Run with Eve", value=False)
compare_qber = st.sidebar.checkbox("Compare QBER with/without Eve", value=False)
channel_noise = st.sidebar.slider("Channel noise (probability of bit flip)", min_value=0.0, max_value=0.2, step=0.01, value=0.0)
distance = st.sidebar.slider("Distance between Alice & Bob (arbitrary units)", min_value=1, max_value=50, step=1, value=10)
reset_experiment = st.sidebar.button("Reset Experiment")
plot_graph = st.sidebar.button("Plot Graph")

# Utility functions
def generate_bb84_keys(num_qubits, noise=0.0, eve=False):
    alice_bits = np.random.randint(2, size=num_qubits)
    alice_basis = np.random.randint(2, size=num_qubits)
    bob_basis = np.random.randint(2, size=num_qubits)
    eve_basis = np.random.randint(2, size=num_qubits) if eve else None

    kept_indices = []
    table_data = []
    for i in range(num_qubits):
        bit = alice_bits[i]
        # Apply channel noise
        if np.random.rand() < noise:
            bit ^= 1
        # Eve interception
        if eve:
            if alice_basis[i] != eve_basis[i]:
                bit ^= 1
        # Bob measures
        measured_bit = bit if alice_basis[i] == bob_basis[i] else np.random.randint(2)
        # Basis match
        basis_match = "Yes" if alice_basis[i] == bob_basis[i] else "No"
        status = "Kept" if basis_match=="Yes" else "Discarded"
        if status=="Kept":
            kept_indices.append(i)
        table_row = {
            "Alice Bit": alice_bits[i],
            "Alice Basis": alice_basis[i],
            "Bob Basis": bob_basis[i],
            "Basis Match": basis_match,
            "Status": status,
            "Bob Bit": measured_bit
        }
        if eve:
            table_row["Eve Basis"] = eve_basis[i]
        table_data.append(table_row)
    # Keys
    alice_key = ''.join([str(alice_bits[i]) for i in kept_indices])
    bob_key = ''.join([str([measured_bit for idx, measured_bit in enumerate([row["Bob Bit"] for row in table_data]) if idx in kept_indices][i]) for i in range(len(kept_indices))])
    # QBER
    qber = np.sum(np.array(list(alice_key)) != np.array(list(bob_key))) / len(alice_key) if len(alice_key)>0 else 0.0
    df = pd.DataFrame(table_data)
    return df, alice_key, bob_key, qber

# Main Experiment
st.subheader("Introduction")
if experiment=="Experiment 1: BB84 with/without Eve":
    st.write("**Aim:** Observe the impact of Eve's presence on BB84 key distribution and QBER.")
    st.write("**Procedure:** Select the number of qubits, choose with/without Eve, click Run. Observe the BB84 process table, keys, and QBER.")

df, alice_key, bob_key, qber = generate_bb84_keys(num_qubits, noise=channel_noise, eve=run_with_eve)

st.subheader("BB84 Protocol Process")
st.dataframe(df)

st.subheader("Generated Keys & QBER")
st.write(f"**Alice Key:** {alice_key}")
st.write(f"**Bob Key:** {bob_key}")
st.write(f"**QBER:** {qber:.2f}")

if plot_graph:
    st.subheader("QBER Comparison Graph")
    qubits_range = [8,16,32,64,128]
    qber_no_eve = []
    qber_with_eve = []
    for nq in qubits_range:
        _, _, _, q1 = generate_bb84_keys(nq, noise=0.0, eve=False)
        _, _, _, q2 = generate_bb84_keys(nq, noise=0.0, eve=True)
        qber_no_eve.append(q1)
        qber_with_eve.append(q2)
    plt.figure(figsize=(8,4))
    plt.plot(qubits_range, qber_no_eve, marker='o', label="Without Eve")
    plt.plot(qubits_range, qber_with_eve, marker='o', label="With Eve")
    plt.xlabel("Number of Qubits")
    plt.ylabel("QBER")
    plt.title("Impact of Number of Qubits on QBER")
    plt.legend()
    st.pyplot(plt)

