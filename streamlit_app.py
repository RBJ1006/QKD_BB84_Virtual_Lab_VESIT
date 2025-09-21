# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

# -------------------------------
# BB84 Quantum Key Distribution Demo
# -------------------------------

st.title("🔑 BB84 Quantum Key Distribution (QKD) Lab")

st.markdown("""
This interactive demo simulates the **BB84 protocol** step by step.
Each qubit transmission is shown in a table.
""")

# User input: number of qubits
n = st.slider("Number of qubits to simulate", min_value=4, max_value=20, value=8)

# Step 1: Alice chooses random bits and bases
alice_bits = np.random.randint(2, size=n)
alice_bases = np.random.randint(2, size=n)

# Step 2: Bob chooses random bases
bob_bases = np.random.randint(2, size=n)

# Step 3: Alice prepares + Bob measures
backend = Aer.get_backend("aer_simulator")
bob_results = []

for i in range(n):
    qc = QuantumCircuit(1, 1)
    if alice_bits[i] == 1:
        qc.x(0)
    if alice_bases[i] == 1:
        qc.h(0)
    if bob_bases[i] == 1:
        qc.h(0)
    qc.measure(0, 0)
    tqc = transpile(qc, backend)
    job = backend.run(tqc, shots=1)
    result = job.result()
    counts = result.get_counts()
    measured_bit = int(list(counts.keys())[0])
    bob_results.append(measured_bit)

bob_results = np.array(bob_results)

# Step 4: Sifting & QBER
mask = alice_bases == bob_bases
alice_key = alice_bits[mask]
bob_key = bob_results[mask]

# QBER calculation
if len(alice_key) > 0:
    errors = np.sum(alice_key != bob_key)
    qber = errors / len(alice_key)
else:
    qber = 0.0

# Step 5: Build Table
rows = []
for i in range(n):
    photon_state = "H/V" if alice_bases[i] == 0 else "+/×"
    basis_match = "Yes" if alice_bases[i] == bob_bases[i] else "No"
    key_bit = alice_bits[i] if basis_match == "Yes" else "-"
    rows.append([
        i+1,
        alice_bits[i],
        "Z" if alice_bases[i] == 0 else "X",
        photon_state,
        bob_bases[i],
        bob_results[i],
        basis_match,
        key_bit
    ])

df = pd.DataFrame(rows, columns=[
    "Qubit #",
    "Alice Bit",
    "Alice Basis",
    "Photon Sent",
    "Bob Basis",
    "Bob Result",
    "Basis Match?",
    "Key Bit"
])

st.subheader("📋 Transmission Table")
st.dataframe(df, use_container_width=True)

st.subheader("🔑 Final Key")
st.write("Alice key:", alice_key)
st.write("Bob key:  ", bob_key)

st.subheader("📊 QBER (Quantum Bit Error Rate)")
st.write(f"{qber*100:.2f}%")

# Optional visualization
if len(alice_key) > 0:
    st.subheader("Key Agreement Plot")
    fig, ax = plt.subplots()
    ax.plot(alice_key, "o-", label="Alice")
    ax.plot(bob_key, "x--", label="Bob")
    ax.set_xlabel("Key index")
    ax.set_ylabel("Bit value")
    ax.legend()
    st.pyplot(fig)

        
