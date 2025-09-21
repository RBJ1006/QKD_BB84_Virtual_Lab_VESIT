# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, execute, Aer

# ------------------------------
# Helper Functions
# ------------------------------

def simulate_bb84(n_qubits=16, eve=False):
    """Simulate BB84 protocol with or without Eve."""
    alice_bits = np.random.randint(2, size=n_qubits)
    alice_bases = np.random.randint(2, size=n_qubits)  # 0=Z, 1=X
    bob_bases = np.random.randint(2, size=n_qubits)

    # Eve variables
    if eve:
        eve_bases = np.random.randint(2, size=n_qubits)
        eve_results = []
    else:
        eve_bases = [None] * n_qubits
        eve_results = [None] * n_qubits

    bob_results = []

    for i in range(n_qubits):
        qc = QuantumCircuit(1,1)
        if alice_bits[i] == 1:
            qc.x(0)
        if alice_bases[i] == 1:
            qc.h(0)

        # Eve intercepts
        if eve:
            if eve_bases[i] == 1:
                qc.h(0)
            qc.measure(0,0)
            sim = Aer.get_backend('qasm_simulator')
            job = execute(qc, backend=sim, shots=1)
            result = job.result()
            counts = result.get_counts()
            eve_bit = int(list(counts.keys())[0])
            eve_results.append(eve_bit)

            # Resend photon to Bob
            qc = QuantumCircuit(1,1)
            if eve_bit == 1:
                qc.x(0)
            if eve_bases[i] == 1:
                qc.h(0)

        # Bob measures
        if bob_bases[i] == 1:
            qc.h(0)
        qc.measure(0,0)
        sim = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend=sim, shots=1)
        result = job.result()
        counts = result.get_counts()
        bob_bit = int(list(counts.keys())[0])
        bob_results.append(bob_bit)

    # Transmission table
    df = pd.DataFrame({
        "Alice Bit": alice_bits,
        "Alice Basis": ["Z" if b==0 else "X" for b in alice_bases],
        "Eve Basis": ["-" if e is None else ("Z" if e==0 else "X") for e in eve_bases],
        "Eve Result": ["-" if e is None else e for e in eve_results],
        "Bob Basis": ["Z" if b==0 else "X" for b in bob_bases],
        "Bob Result": bob_results
    })

    # Key sifting
    mask = alice_bases == bob_bases
    alice_key = alice_bits[mask]
    bob_key = np.array(bob_results)[mask]

    # QBER calculation
    if len(alice_key) > 0:
        qber = np.sum(alice_key != bob_key) / len(alice_key)
    else:
        qber = 0.0

    return df, alice_key, bob_key, qber

def display_keys(alice_key, bob_key):
    st.write("### 🔑 Alice & Bob Keys (after sifting)")
    st.write(f"Alice Key: {''.join(map(str, alice_key))}")
    st.write(f"Bob Key:   {''.join(map(str, bob_key))}")

def plot_qber_comparison(qber_no_eve, qber_with_eve):
    st.write("### 📊 QBER Comparison")
    fig, ax = plt.subplots()
    ax.bar(["Without Eve", "With Eve"], [qber_no_eve, qber_with_eve], color=["green","red"])
    ax.set_ylabel("QBER")
    st.pyplot(fig)

def plot_qber_vs_qubits(results_dict, title):
    st.write(f"### 📈 {title}")
    fig, ax = plt.subplots()
    ax.plot(list(results_dict.keys()), list(results_dict.values()), marker="o")
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("QBER")
    st.pyplot(fig)

# ------------------------------
# Streamlit Layout
# ------------------------------

st.set_page_config(page_title="QKD BB84 Virtual Lab", layout="wide")
st.title("🔐 Quantum Key Distribution (BB84 Protocol) Virtual Lab")

# Sidebar
st.sidebar.header("⚙️ Experiment Controls")
n_qubits = st.sidebar.selectbox("Select number of qubits", [8,16,32,64,128], index=1)
reset = st.sidebar.button("🔄 Reset Experiment")

if "results" not in st.session_state or reset:
    st.session_state.results = {}
    st.session_state.results_eve = {}

# Run Without Eve
if st.sidebar.button("▶️ Run Without Eve"):
    df, alice_key, bob_key, qber = simulate_bb84(n_qubits, eve=False)
    st.write("### 📋 Transmission Table (Without Eve)")
    st.dataframe(df)
    display_keys(alice_key, bob_key)
    st.write(f"**QBER (Without Eve):** {qber:.2%}")
    st.session_state.results[n_qubits] = qber

# Run With Eve
if st.sidebar.button("▶️ Run With Eve"):
    df, alice_key, bob_key, qber = simulate_bb84(n_qubits, eve=True)
    st.write("### 📋 Transmission Table (With Eve)")
    st.dataframe(df)
    display_keys(alice_key, bob_key)
    st.write(f"**QBER (With Eve):** {qber:.2%}")
    st.session_state.results_eve[n_qubits] = qber

# Compare Eve vs No Eve
if st.sidebar.button("📊 Compare Eve vs No Eve"):
    qber_no_eve = np.mean(list(st.session_state.results.values())) if st.session_state.results else 0
    qber_with_eve = np.mean(list(st.session_state.results_eve.values())) if st.session_state.results_eve else 0
    plot_qber_comparison(qber_no_eve, qber_with_eve)

# Plot QBER vs Qubits
if st.sidebar.button("📈 Plot QBER vs Qubits"):
    if st.session_state.results:
        plot_qber_vs_qubits(st.session_state.results, "QBER vs Qubits (Without Eve)")
    if st.session_state.results_eve:
        plot_qber_vs_qubits(st.session_state.results_eve, "QBER vs Qubits (With Eve)")


    
       
    
  
