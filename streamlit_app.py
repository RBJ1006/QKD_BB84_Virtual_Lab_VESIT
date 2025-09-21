# Paste the full streamlit_app.py code here
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

st.set_page_config(page_title="QXplore: Quantum Key Distribution (BB84 Protocol) Virtual Lab",
                   layout="wide")

st.title("QXplore: Quantum Key Distribution (BB84 Protocol) Virtual Lab")

# --------------------------- Helper functions ---------------------------
def generate_bb84_bits(num_qubits):
    alice_bits = np.random.randint(2, size=num_qubits)
    alice_bases = np.random.randint(2, size=num_qubits)
    bob_bases = np.random.randint(2, size=num_qubits)
    return alice_bits, alice_bases, bob_bases

def simulate_bb84(alice_bits, alice_bases, bob_bases, eve=False, noise=0.0):
    num_qubits = len(alice_bits)
    key_bits = []
    table = []

    for i in range(num_qubits):
        bit = alice_bits[i]
        basis = alice_bases[i]
        bob_basis = bob_bases[i]

        # Eve interception
        if eve:
            eve_basis = np.random.randint(2)
            if eve_basis != basis:
                # Eve disturbs the photon
                bit = np.random.randint(2)
        else:
            eve_basis = None

        # Noise in channel
        if np.random.rand() < noise:
            bit = 1 - bit  # flip

        basis_match = basis == bob_basis
        status = "Kept" if basis_match else "Discarded"
        if basis_match:
            key_bits.append(bit)

        table.append([i+1, bit, basis, bob_basis, "Yes" if basis_match else "No",
                      status, eve_basis])

    key_bits = np.array(key_bits)
    return table, key_bits

def compute_qber(alice_key, bob_key):
    min_len = min(len(alice_key), len(bob_key))
    if min_len == 0:
        return 0
    errors = np.sum(alice_key[:min_len] != bob_key[:min_len])
    return errors / min_len

# --------------------------- Sidebar: Experiment selection ---------------------------
exp_choice = st.sidebar.selectbox("Select Experiment",
                                  ["1. BB84 with/without Eve",
                                   "2. Qubit variation impact",
                                   "3. Channel noise impact",
                                   "4. Distance impact"])

st.sidebar.markdown("---")

# --------------------------- Common Controls ---------------------------
num_qubits = st.sidebar.slider("Number of Qubits", min_value=8, max_value=128, value=16, step=8)
noise = st.sidebar.slider("Channel Noise (0.0 - 1.0)", min_value=0.0, max_value=0.2, value=0.0, step=0.01)
eve_present = st.sidebar.checkbox("Include Eve in simulation", value=False)
reset_btn = st.sidebar.button("Reset / Rerun Simulation")

# --------------------------- Experiment 1: BB84 with/without Eve ---------------------------
if exp_choice.startswith("1"):
    st.subheader("Experiment 1: BB84 Protocol with/without Eve")
    st.markdown("""
    **Aim**: Observe the impact of the presence or absence of an eavesdropper (Eve) on key generation and QBER.
    **Procedure**: Select the number of qubits, enable/disable Eve, then run the simulation. The BB84 protocol table shows Alice & Bob bits, basis selection, basis match, and whether the bit is kept or discarded.
    """)
    if reset_btn:
        alice_bits, alice_bases, bob_bases = generate_bb84_bits(num_qubits)
        table, alice_key = simulate_bb84(alice_bits, alice_bases, bob_bases, eve=eve_present, noise=noise)
        df_table = pd.DataFrame(table, columns=["Qubit #", "Alice Bit", "Alice Basis",
                                                "Bob Basis", "Basis Match", "Status", "Eve Basis"])
        st.markdown("**BB84 Protocol Process**")
        st.dataframe(df_table)

        bob_key = alice_key.copy()  # Since no errors simulated apart from Eve/noise
        qber = compute_qber(alice_key, bob_key)
        st.write(f"**Alice Key:** {alice_key}")
        st.write(f"**Bob Key:** {bob_key}")
        st.write(f"**QBER:** {qber:.2f}")

# --------------------------- Experiment 2: Qubit variation impact ---------------------------
elif exp_choice.startswith("2"):
    st.subheader("Experiment 2: Impact of Qubit Number on QBER")
    st.markdown("""
    **Aim**: As the number of qubits changes, observe its impact on QBER.
    **Procedure**: Select number of qubits, optionally include Eve, and run the simulation. Observe how key length and QBER vary.
    """)
    qubit_list = list(range(8, 129, 8))
    qber_list = []
    key_table = []

    for nq in qubit_list:
        alice_bits, alice_bases, bob_bases = generate_bb84_bits(nq)
        table, alice_key = simulate_bb84(alice_bits, alice_bases, bob_bases, eve=eve_present, noise=noise)
        bob_key = alice_key.copy()
        qber_val = compute_qber(alice_key, bob_key)
        qber_list.append(qber_val)
        key_table.append([nq, "".join(map(str, alice_key)), f"{qber_val:.2f}"])

    df_keys = pd.DataFrame(key_table, columns=["No. of Qubits", "Key Generated", "QBER"])
    st.dataframe(df_keys)

    fig, ax = plt.subplots()
    ax.plot(df_keys["No. of Qubits"], df_keys["QBER"], marker="o", color="blue")
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("QBER")
    ax.set_title("Impact of Qubit Variation on QBER")
    st.pyplot(fig)

# --------------------------- Experiment 3: Channel noise impact ---------------------------
elif exp_choice.startswith("3"):
    st.subheader("Experiment 3: Impact of Channel Noise on QBER")
    st.markdown("""
    **Aim**: Observe the effect of channel noise on QBER. Noise represents the probability that a transmitted qubit flips during transmission (e.g., 0.01 = 1% chance of flipping).
    **Procedure**: Select channel noise level, number of qubits, optionally include Eve, and run the simulation.
    """)
    noise_values = np.arange(0.0, 0.21, 0.02)
    qber_list = []
    for n in noise_values:
        alice_bits, alice_bases, bob_bases = generate_bb84_bits(num_qubits)
        table, alice_key = simulate_bb84(alice_bits, alice_bases, bob_bases, eve=eve_present, noise=n)
        bob_key = alice_key.copy()
        qber_val = compute_qber(alice_key, bob_key)
        qber_list.append(qber_val)

    df_noise = pd.DataFrame(list(zip(noise_values, qber_list)), columns=["Channel Noise", "QBER"])
    st.dataframe(df_noise)

    fig, ax = plt.subplots()
    ax.plot(df_noise["Channel Noise"], df_noise["QBER"], marker="o", color="red")
    ax.set_xlabel("Channel Noise")
    ax.set_ylabel("QBER")
    ax.set_title("Impact of Channel Noise on QBER")
    st.pyplot(fig)

# --------------------------- Experiment 4: Distance impact (abstracted) ---------------------------
elif exp_choice.startswith("4"):
    st.subheader("Experiment 4: Impact of Distance on QBER")
    st.markdown("""
    **Aim**: Observe the effect of distance on QBER. Longer distance increases the probability of qubit loss or error.
    **Procedure**: Select an abstract distance factor (simulated via noise), number of qubits, optionally include Eve, and run the simulation.
    """)
    distance_values = np.arange(1, 11, 1)  # arbitrary distance units
    qber_list = []
    for d in distance_values:
        alice_bits, alice_bases, bob_bases = generate_bb84_bits(num_qubits)
        # assume distance introduces noise proportional to distance
        table, alice_key = simulate_bb84(alice_bits, alice_bases, bob_bases, eve=eve_present, noise=0.01*d)
        bob_key = alice_key.copy()
        qber_val = compute_qber(alice_key, bob_key)
        qber_list.append(qber_val)

    df_distance = pd.DataFrame(list(zip(distance_values, qber_list)), columns=["Distance (units)", "QBER"])
    st.dataframe(df_distance)

    fig, ax = plt.subplots()
    ax.plot(df_distance["Distance (units)"], df_distance["QBER"], marker="o", color="green")
    ax.set_xlabel("Distance")
    ax.set_ylabel("QBER")
    ax.set_title("Impact of Distance on QBER")
    st.pyplot(fig)
