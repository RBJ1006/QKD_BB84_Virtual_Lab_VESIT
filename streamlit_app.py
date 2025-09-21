# Paste the full streamlit_app.py code here
# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ------------------------
# Utility functions
# ------------------------

def run_bb84(num_qubits=10, noise_level=0.0, distance_km=0):
    """
    Simulate a BB84-like process: random bases, matches, and key generation.
    Returns kept bits, discarded bits, and QBER.
    """
    alice_bits = np.random.randint(2, size=num_qubits)
    alice_bases = np.random.randint(2, size=num_qubits)
    bob_bases = np.random.randint(2, size=num_qubits)

    kept_bits = []
    discarded_bits = []
    errors = 0

    for i in range(num_qubits):
        bit = alice_bits[i]
        a_basis = alice_bases[i]
        b_basis = bob_bases[i]

        # If bases match, keep
        if a_basis == b_basis:
            # Noise: flip with probability = noise_level
            if np.random.rand() < noise_level:
                measured_bit = 1 - bit
            else:
                measured_bit = bit
            kept_bits.append((i, bit, measured_bit, "Kept"))
            if measured_bit != bit:
                errors += 1
        else:
            discarded_bits.append((i, bit, None, "Discarded"))

    n_kept = len(kept_bits)
    qber = errors / n_kept if n_kept > 0 else 0
    return kept_bits, discarded_bits, qber


# ------------------------
# Streamlit Layout
# ------------------------

st.set_page_config(page_title="QXplore: Quantum Key Distribution (BB84 Protocol) Virtual Lab",
                   layout="wide")

st.title("QXplore: Quantum Key Distribution (BB84 Protocol) Virtual Lab")

experiment = st.sidebar.selectbox(
    "Choose Experiment",
    [
        "Experiment 1: BB84 Protocol Process",
        "Experiment 2: Impact of Qubit Variation on QBER",
        "Experiment 3: Impact of Channel Noise on QBER",
        "Experiment 4: Impact of Distance on QBER"
    ]
)

# ------------------------
# Experiment 1
# ------------------------
if experiment == "Experiment 1: BB84 Protocol Process":
    st.header("Experiment 1: BB84 Protocol Process")
    st.markdown("""
    In this experiment, we demonstrate the **step-by-step process of the BB84 Protocol**:
    - Alice randomly generates bits and bases.
    - Bob randomly chooses measurement bases.
    - If bases match → bit is **kept**.
    - If bases don’t match → bit is **discarded**.
    - QBER (Quantum Bit Error Rate) is calculated as the ratio of errors to kept bits.
    """)

    num_qubits = st.slider("Number of Qubits", 5, 50, 10)
    noise = st.slider("Channel Noise (probability of bit flip)", 0.0, 0.2, 0.01, step=0.01)

    kept, discarded, qber = run_bb84(num_qubits=num_qubits, noise_level=noise)

    df_kept = pd.DataFrame(kept, columns=["Qubit Index", "Alice Bit", "Bob Bit", "Status"])
    df_discarded = pd.DataFrame(discarded, columns=["Qubit Index", "Alice Bit", "Bob Bit", "Status"])

    st.subheader("Kept Bits")
    st.dataframe(df_kept)
    st.subheader("Discarded Bits")
    st.dataframe(df_discarded)

    st.metric("QBER", f"{qber:.2%}")


# ------------------------
# Experiment 2
# ------------------------
elif experiment == "Experiment 2: Impact of Qubit Variation on QBER":
    st.header("Experiment 2: Impact of Qubit Variation on QBER")
    st.markdown("""
    In this experiment, we vary the **number of qubits** transmitted and observe:
    - How many key bits are generated.
    - The resulting QBER.
    """)

    qubit_range = st.slider("Maximum Qubits", 10, 200, 50, step=10)
    noise = st.slider("Channel Noise", 0.0, 0.2, 0.01, step=0.01)

    results = []
    for n in range(10, qubit_range + 1, 10):
        kept, _, qber = run_bb84(num_qubits=n, noise_level=noise)
        results.append((n, len(kept), qber))

    df = pd.DataFrame(results, columns=["No. of Qubits", "Key Generated (bits)", "QBER"])
    st.dataframe(df)

    # Plot
    fig, ax1 = plt.subplots()
    ax1.plot(df["No. of Qubits"], df["QBER"], marker="o")
    ax1.set_xlabel("No. of Qubits")
    ax1.set_ylabel("QBER", color="red")
    st.pyplot(fig)


# ------------------------
# Experiment 3
# ------------------------
elif experiment == "Experiment 3: Impact of Channel Noise on QBER":
    st.header("Experiment 3: Impact of Channel Noise on QBER")
    st.markdown("""
    In this experiment, we vary the **channel noise level** (probability of bit flip during transmission).
    - Higher noise leads to higher QBER.
    - QBER = Errors / Kept bits.
    """)

    num_qubits = st.slider("Number of Qubits", 50, 200, 100, step=10)

    noise_levels = np.linspace(0, 0.2, 10)
    results = []
    for n in noise_levels:
        kept, _, qber = run_bb84(num_qubits=num_qubits, noise_level=n)
        results.append((round(n, 3), len(kept), qber))

    df = pd.DataFrame(results, columns=["Channel Noise (p)", "Key Generated (bits)", "QBER"])
    st.dataframe(df)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(df["Channel Noise (p)"], df["QBER"], marker="s")
    ax.set_xlabel("Channel Noise Probability")
    ax.set_ylabel("QBER")
    st.pyplot(fig)


# ------------------------
# Experiment 4
# ------------------------
elif experiment == "Experiment 4: Impact of Distance on QBER":
    st.header("Experiment 4: Impact of Distance on QBER")
    st.markdown("""
    In this experiment, we vary the **distance between Alice and Bob**.
    - Greater distance usually leads to higher noise and error probability.
    - For simplicity, we model error probability as `noise = distance_km * 0.0005`.
    """)

    num_qubits = st.slider("Number of Qubits", 50, 200, 100, step=10)
    max_distance = st.slider("Max Distance (km)", 10, 500, 100, step=10)

    results = []
    for d in range(0, max_distance + 1, 20):
        noise = d * 0.0005
        kept, _, qber = run_bb84(num_qubits=num_qubits, noise_level=noise)
        results.append((d, len(kept), qber))

    df = pd.DataFrame(results, columns=["Distance (km)", "Key Generated (bits)", "QBER"])
    st.dataframe(df)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(df["Distance (km)"], df["QBER"], marker="^")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("QBER")
    st.pyplot(fig)


       
       

    
    
   
   
