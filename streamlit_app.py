# Paste the full streamlit_app.py code here
# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

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
channel_noise = st.sidebar.slider("Channel noise (0-1)", min_value=0.0, max_value=0.2, step=0.01, value=0.0)
distance = st.sidebar.slider("Distance (arbitrary units)", min_value=1, max_value=50, step=1, value=10)
reset_experiment = st.sidebar.button("Reset Experiment")
plot_graph = st.sidebar.button("Plot Graph")

# ----------------------------------------
# Utility function: BB84 key generation
# ----------------------------------------
def generate_bb84_keys(num_qubits, noise=0.0, eve=False):
    alice_bits = np.random.randint(2, size=num_qubits)
    alice_basis = np.random.randint(2, size=num_qubits)
    bob_basis = np.random.randint(2, size=num_qubits)
    eve_basis = np.random.randint(2, size=num_qubits) if eve else None

    kept_indices = []
    table_data = []
    for i in range(num_qubits):
        bit = alice_bits[i]
        if np.random.rand() < noise:  # channel noise
            bit ^= 1
        if eve:  # Eve interception
            if alice_basis[i] != eve_basis[i]:
                bit ^= 1
        # Bob measures
        measured_bit = bit if alice_basis[i] == bob_basis[i] else np.random.randint(2)
        basis_match = "Yes" if alice_basis[i] == bob_basis[i] else "No"
        status = "Kept" if basis_match=="Yes" else "Discarded"
        if status=="Kept":
            kept_indices.append(i)
        row = {
            "Alice Bit": alice_bits[i],
            "Alice Basis": alice_basis[i],
            "Bob Basis": bob_basis[i],
            "Basis Match": basis_match,
            "Status": status,
            "Bob Bit": measured_bit
        }
        if eve:
            row["Eve Basis"] = eve_basis[i]
        table_data.append(row)
    
    alice_key = ''.join([str(alice_bits[i]) for i in kept_indices])
    bob_key = ''.join([str(table_data[i]["Bob Bit"]) for i in kept_indices])
    qber = np.sum(np.array(list(alice_key)) != np.array(list(bob_key))) / len(alice_key) if len(alice_key)>0 else 0.0
    df = pd.DataFrame(table_data)
    return df, alice_key, bob_key, qber

# ----------------------------------------
# Experiment Introduction
# ----------------------------------------
st.subheader("Introduction")
if experiment=="Experiment 1: BB84 with/without Eve":
    st.write("**Aim:** Observe the impact of Eve's presence on BB84 key distribution and QBER.")
    st.write("**Procedure:** Select number of qubits, choose with/without Eve, click Run. Observe the BB84 process table, keys, and QBER.")

elif experiment=="Experiment 2: Qubit Variation Impact on QBER":
    st.write("**Aim:** Study how the number of qubits affects the QBER in BB84 protocol, with and without Eve.")
    st.write("**Procedure:** Increase number of qubits, click Run. Observe keys, QBER, and table updates. Plot QBER vs Number of Qubits.")

elif experiment=="Experiment 3: Channel Noise Impact on QBER":
    st.write("**Aim:** Study how channel noise affects QBER in BB84 protocol, with and without Eve.")
    st.write("**Procedure:** Adjust channel noise probability, click Run. Observe keys, QBER, and dynamic table. Plot QBER vs Channel Noise.")

elif experiment=="Experiment 4: Distance Impact on QBER":
    st.write("**Aim:** Study how distance between Alice and Bob affects QBER in BB84 protocol.")
    st.write("**Procedure:** Adjust distance, click Run. Observe keys, QBER, and dynamic table. Plot QBER vs Distance.")

# ----------------------------------------
# Run the simulation
# ----------------------------------------
df, alice_key, bob_key, qber = generate_bb84_keys(num_qubits, noise=channel_noise, eve=run_with_eve)

st.subheader("BB84 Protocol Process")
st.dataframe(df)

st.subheader("Generated Keys & QBER")
st.write(f"**Alice Key:** {alice_key}")
st.write(f"**Bob Key:** {bob_key}")
st.write(f"**QBER:** {qber:.2f}")

# ----------------------------------------
# Plotting
# ----------------------------------------
if plot_graph:
    st.subheader("Comparison Graph")
    param_values = []
    qber_no_eve = []
    qber_with_eve = []
    if experiment=="Experiment 2: Qubit Variation Impact on QBER":
        param_values = [8,16,32,64,128]
        for nq in param_values:
            _, _, _, q1 = generate_bb84_keys(nq, noise=0.0, eve=False)
            _, _, _, q2 = generate_bb84_keys(nq, noise=0.0, eve=True)
            qber_no_eve.append(q1)
            qber_with_eve.append(q2)
        plt.xlabel("Number of Qubits")
        plt.ylabel("QBER")
        plt.title("Impact of Number of Qubits on QBER")

    elif experiment=="Experiment 3: Channel Noise Impact on QBER":
        param_values = np.arange(0.0, 0.21, 0.02)
        for noise in param_values:
            _, _, _, q1 = generate_bb84_keys(num_qubits, noise=noise, eve=False)
            _, _, _, q2 = generate_bb84_keys(num_qubits, noise=noise, eve=True)
            qber_no_eve.append(q1)
            qber_with_eve.append(q2)
        plt.xlabel("Channel Noise Probability")
        plt.ylabel("QBER")
        plt.title("Impact of Channel Noise on QBER")

    elif experiment=="Experiment 4: Distance Impact on QBER":
        param_values = np.arange(1, 51, 5)
        for dist in param_values:
            # For simplicity, distance effect simulated as noise proportional to distance
            noise_effect = dist*0.002
            _, _, _, q1 = generate_bb84_keys(num_qubits, noise=noise_effect, eve=False)
            _, _, _, q2 = generate_bb84_keys(num_qubits, noise=noise_effect, eve=True)
            qber_no_eve.append(q1)
            qber_with_eve.append(q2)
        plt.xlabel("Distance (arbitrary units)")
        plt.ylabel("QBER")
        plt.title("Impact of Distance on QBER")

    plt.plot(param_values, qber_no_eve, marker='o', label="Without Eve")
    plt.plot(param_values, qber_with_eve, marker='o', label="With Eve")
    plt.legend()
    st.pyplot(plt)


        
