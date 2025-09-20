# Paste your full app code here
import streamlit as st
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator

st.set_page_config(page_title="QKD Virtual Lab - BB84 Protocol", layout="wide")

# (Rest of your BB84 app code goes here)

# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd

# Qiskit imports using AerSimulator for Python 3.13 compatibility
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator

# Set page configuration
st.set_page_config(
    page_title="QKD Virtual Lab - BB84 Protocol",
    layout="wide"
)

# --- Sidebar ---
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
        # Step 1: Random bits and bases
        alice_bits = np.random.randint(2, size=n_bits)
        alice_bases = np.random.randint(2, size=n_bits)
        bob_bases = np.random.randint(2, size=n_bits)
        
        # Step 2: Qiskit simulation using AerSimulator
        key_bits = []
        table_data = []

        simulator = AerSimulator()

        for i in range(n_bits):
            qc = QuantumCircuit(1, 1)
            
            # Encode bit
            if alice_bits[i] == 1:
                qc.x(0)
            if alice_bases[i] == 1:
                qc.h(0)
            
            # Eve intervention
            if eavesdrop:
                eve_basis = np.random.randint(2)
                if eve_basis == 1:
                    qc.h(0)
                # Measure and reset
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
            
            # Only keep bits where bases match
            if alice_bases[i] == bob_bases[i]:
                key_bits.append(meas_bit)
            
            table_data.append([i+1, alice_bits[i], alice_bases[i], bob_bases[i], meas_bit])

        # Display table
        df = pd.DataFrame(table_data, columns=["Qubit#", "Alice Bit", "Alice Basis", "Bob Basis", "Bob Measured"])
        st.subheader("Step-by-step Measurement Table")
        st.dataframe(df)

        # Show final key
        st.subheader("Final Shared Key")
        st.code("".join(map(str, key_bits)))
