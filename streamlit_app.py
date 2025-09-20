import streamlit as st
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, Aer, execute

st.set_page_config(page_title="QKD Virtual Lab - BB84 Protocol", layout="wide")

# Title
st.title("🔑 Quantum Key Distribution Virtual Lab - BB84 Protocol")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Introduction", "Simulation"])

# -----------------------------
# 1. INTRODUCTION PAGE
# -----------------------------
if page == "Introduction":
    st.header("Why Quantum Key Distribution (QKD)?")
    st.markdown(
        """
        - Classical cryptography relies on mathematical complexity (e.g., RSA, ECC).  
        - Quantum computers threaten these systems.  
        - QKD provides **information-theoretic security** based on quantum mechanics.  

        ### Classification of QKD Protocols:
        - **Prepare-and-measure protocols**: e.g., BB84, B92.  
        - **Entanglement-based protocols**: e.g., E91.  
        - **Measurement-device-independent QKD (MDI-QKD)**.  

        ### The BB84 Protocol (1984)
        - Alice encodes random bits using random bases (rectilinear or diagonal).  
        - Bob measures in random bases.  
        - After basis reconciliation, they keep only bits where bases match.  
        - An eavesdropper (Eve) introduces detectable errors.  
        """)
    st.info("👉 Switch to the **Simulation** tab to try the BB84 protocol.")

# -----------------------------
# 2. SIMULATION PAGE
# -----------------------------
elif page == "Simulation":
    st.header("BB84 Protocol Simulation")

    st.sidebar.subheader("Simulation Controls")
    num_qubits = st.sidebar.slider("Number of qubits", 4, 32, 8, step=2)
    add_eve = st.sidebar.checkbox("Include Eve (eavesdropper)?", value=False)

    st.markdown("### Step 1: Alice generates random bits and bases")
    alice_bits = np.random.randint(2, size=num_qubits)
    alice_bases = np.random.randint(2, size=num_qubits)
    st.write("Alice's bits:", alice_bits)
    st.write("Alice's bases (0=Z, 1=X):", alice_bases)

    st.markdown("### Step 2: Bob chooses random bases")
    bob_bases = np.random.randint(2, size=num_qubits)
    st.write("Bob's bases (0=Z, 1=X):", bob_bases)

    # Quantum simulation
    backend = Aer.get_backend("qasm_simulator")
    bob_results = []

    for i in range(num_qubits):
        qc = QuantumCircuit(1, 1)
        # Alice encodes qubit
        if alice_bits[i] == 1:
            qc.x(0)
        if alice_bases[i] == 1:
            qc.h(0)

        # Eve (optional)
        if add_eve:
            eve_basis = np.random.randint(2)
            if eve_basis == 1:
                qc.h(0)
            qc.measure(0, 0)
            job = execute(qc, backend, shots=1)
            eve_result = int(list(job.result().get_counts().keys())[0])
            qc = QuantumCircuit(1, 1)
            if eve_result == 1:
                qc.x(0)
            if eve_basis == 1:
                qc.h(0)

        # Bob's measurement
        if bob_bases[i] == 1:
            qc.h(0)
        qc.measure(0, 0)
        job = execute(qc, backend, shots=1)
        result = int(list(job.result().get_counts().keys())[0])
        bob_results.append(result)

    st.markdown("### Step 3: Measurement results")
    st.write("Bob's results:", bob_results)

    # Step 4: Basis reconciliation
    mask = alice_bases == bob_bases
    sifted_alice = alice_bits[mask]
    sifted_bob = np.array(bob_results)[mask]

    st.markdown("### Step 4: Basis reconciliation (keeping only matching bases)")
    df = pd.DataFrame({
        "Alice bit": alice_bits,
        "Alice basis": alice_bases,
        "Bob basis": bob_bases,
        "Bob result": bob_results,
        "Keep?": mask
    })
    st.dataframe(df)

    st.markdown("### Step 5: Final Key")
    st.success(f"Alice's sifted key: {sifted_alice}")
    st.success(f"Bob's sifted key:   {sifted_bob}")

    # Error rate
    if len(sifted_alice) > 0:
        qber = np.sum(sifted_alice != sifted_bob) / len(sifted_alice)
        st.warning(f"Quantum Bit Error Rate (QBER): {qber:.2%}")
