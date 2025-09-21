# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram

# -------------------------------
# BB84 Quantum Key Distribution Demo
# -------------------------------

st.title("🔑 BB84 Quantum Key Distribution (QKD) Lab")

st.markdown("""
This is a simple interactive demo of the **BB84 protocol**  
implemented using **Qiskit + Streamlit**.
""")

# User input: number of qubits
n = st.slider("Number of qubits to simulate", min_value=4, max_value=20, value=8)

# Step 1: Alice chooses random bits and bases
alice_bits = np.random.randint(2, size=n)
alice_bases = np.random.randint(2, size=n)

# Step 2: Bob chooses random bases
bob_bases = np.random.randint(2, size=n)

st.subheader("Alice's random choices")
st.write("Bits:   ", alice_bits)
st.write("Bases:  ", alice_bases)

st.subheader("Bob's random bases")
st.write("Bases:  ", bob_bases)

# Step 3: Alice prepares qubits
circuits = []
for i in range(n):
    qc = QuantumCircuit(1, 1)
    if alice_bits[i] == 1:
        qc.x(0)
    if alice_bases[i] == 1:
        qc.h(0)
    # Bob’s measurement
    if bob_bases[i] == 1:
        qc.h(0)
    qc.measure(0, 0)
    circuits.append(qc)

# Step 4: Run on simulator
backend = Aer.get_backend("aer_simulator")
results = []
for qc in circuits:
    tqc = transpile(qc, backend)
    job = backend.run(tqc, shots=1)
    result = job.result()
    counts = result.get_counts()
    measured_bit = int(list(counts.keys())[0])
    results.append(measured_bit)

bob_results = np.array(results)

st.subheader("Bob's measurement results")
st.write(bob_results)

# Step 5: Sifting
mask = alice_bases == bob_bases
alice_key = alice_bits[mask]
bob_key = bob_results[mask]

st.subheader("Sifted Key")
st.write("Alice key:", alice_key)
st.write("Bob key:  ", bob_key)

# Visualization
st.subheader("Key Agreement")
fig, ax = plt.subplots()
ax.plot(alice_key, "o-", label="Alice")
ax.plot(bob_key, "x--", label="Bob")
ax.set_xlabel("Key index")
ax.set_ylabel("Bit value")
ax.legend()
st.pyplot(fig)

   
        
