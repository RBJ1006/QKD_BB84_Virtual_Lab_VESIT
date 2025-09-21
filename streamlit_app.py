# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator  # AerSimulator is now in providers.aer
from qiskit.utils import QuantumInstance
from qiskit.execute_function import execute  # execute is now imported from execute_function


# Set up Streamlit page
st.set_page_config(page_title="QKD Virtual Lab - BB84 Protocol", layout="wide")

# Sidebar navigation
st.sidebar.title("BB84 Virtual Lab")
tab = st.sidebar.radio("Go to", ["Introduction", "Simulation"])

# --- Introduction Tab ---
if tab == "Introduction":
    st.title("Quantum Key Distribution: BB84 Protocol")
    st.markdown("""
    **Why QKD?**
    Quantum Key Distribution allows two parties to share a secure key using quantum mechanics.

    **BB84 Protocol Overview:**
    - Sender (Alice) encodes bits using random bases (Z or X).
    - Receiver (Bob) measures in random bases.
    - They compare bases over a public channel to create a shared key.
    - Presence of eavesdropper (Eve) can be detected due to quantum disturbance.
    """)

# --- Simulation Tab ---
if tab == "Simulation":
    st.title("BB84 Simulation")

    n_bits = st.slider("Number of qubits (photons)", min_value=4, max_value=20, value=8, step=1)
    eavesdrop = st.checkbox("Include eavesdropper (Eve)?", value=False)

    if st.button("Run Simulation"):
        # Initialize
        alice_bits = np.random.randint(2, size=n_bits)
        alice_bases = np.random.randint(2, size=n_bits)
        bob_bases = np.random.randint(2, size=n_bits)

        key_bits = []
        table_data = []
        simulator = Aer.get_backend('aer_simulator')  # AerSimulator works in Python 3.11

        for i in range(n_bits):
            qc = QuantumCircuit(1, 1)

            # Alice encodes
            if alice_bits[i] == 1:
                qc.x(0)
            if alice_bases[i] == 1:
                qc.h(0)

            # Eve interference
            if eavesdrop:
                eve_basis = np.random.randint(2)
                if eve_basis == 1:
                    qc.h(0)
                qc.measure(0, 0)
                result = execute(qc, simulator, shots=1).result()
                meas = int(list(result.get_counts().keys())[0])
                qc.reset(0)
                if meas == 1:
                    qc.x(0)
                if eve_basis == 1:
                    qc.h(0)
                qc.barrier()

            # Bob measurement
            if bob_bases[i] == 1:
                qc.h(0)
            qc.measure(0, 0)

            result = execute(qc, simulator, shots=1).result()
            meas_bit = int(list(result.get_counts().keys())[0])

            # Keep key bits where bases match
            if alice_bases[i] == bob_bases[i]:
                key_bits.append(meas_bit)

            table_data.append([i+1, alice_bits[i], alice_bases[i], bob_bases[i], meas_bit])

        # Display table
        df = pd.DataFrame(table_data, columns=["Qubit#", "Alice Bit", "Alice Basis", "Bob Basis", "Bob Measured"])
        st.subheader("Step-by-step Measurement Table")
        st.dataframe(df)

        # Display final shared key
        st.subheader("Final Shared Key")
        st.code("".join(map(str, key_bits)))
