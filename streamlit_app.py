# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute

st.set_page_config(
    page_title="QXplore: Quantum Key Distribution (BB84 Protocol) Virtual Lab",
    layout="wide"
)

st.title("QXplore: Quantum Key Distribution (BB84 Protocol) Virtual Lab")

# Sidebar for experiment selection
experiment = st.sidebar.selectbox(
    "Select Experiment",
    [
        "Experiment 1: BB84 with/without Eve",
        "Experiment 2: BB84 by varying qubits",
        "Experiment 3: Impact of channel noise",
        "Experiment 4: Impact of distance"
    ]
)

# Utility functions
def generate_bits(n):
    return np.random.randint(0, 2, n)

def generate_bases(n):
    return np.random.randint(0, 2, n)

def bb84_simulation(num_qubits=8, eve=False, noise=0.0):
    alice_bits = generate_bits(num_qubits)
    alice_bases = generate_bases(num_qubits)
    bob_bases = generate_bases(num_qubits)
    
    eve_bases = generate_bases(num_qubits) if eve else None
    eve_bits = alice_bits.copy() if eve else None

    transmitted_bits = []
    final_key_indices = []

    for i in range(num_qubits):
        # Eve interception
        if eve:
            if eve_bases[i] != alice_bases[i]:
                # Eve measures in wrong basis, collapses qubit
                eve_bits[i] = np.random.randint(0, 2)

        # Bob measurement
        if eve:
            if bob_bases[i] == (eve_bases[i] if eve else alice_bases[i]):
                measured_bit = eve_bits[i]
            else:
                measured_bit = np.random.randint(0, 2)
        else:
            if bob_bases[i] == alice_bases[i]:
                measured_bit = alice_bits[i]
            else:
                measured_bit = np.random.randint(0, 2)
        
        # Apply channel noise
        if np.random.rand() < noise:
            measured_bit = 1 - measured_bit

        transmitted_bits.append(measured_bit)

    # Determine kept bits
    basis_match = [("Yes" if alice_bases[i]==bob_bases[i] else "No") for i in range(num_qubits)]
    status = [("Kept" if bm=="Yes" else "Discarded") for bm in basis_match]
    final_key_indices = [i for i,bm in enumerate(basis_match) if bm=="Yes"]

    alice_key = [alice_bits[i] for i in final_key_indices]
    bob_key = [transmitted_bits[i] for i in final_key_indices]

    # QBER calculation
    if len(alice_key) > 0:
        qber = np.sum(np.array(alice_key) != np.array(bob_key)) / len(alice_key)
    else:
        qber = 0

    # Table
    table_data = {
        "Alice Bit": alice_bits,
        "Alice Basis": alice_bases,
        "Bob Basis": bob_bases,
        "Basis Match": basis_match,
        "Status": status,
        "Bob Bit": transmitted_bits
    }
    if eve:
        table_data["Eve Basis"] = eve_bases
        table_data["Eve Bit"] = eve_bits

    df = pd.DataFrame(table_data)

    return df, alice_key, bob_key, qber

# ----------- EXPERIMENTS ----------------
if experiment == "Experiment 1: BB84 with/without Eve":
    st.header("Experiment 1: Study of BB84 protocol with Eve & Without Eve")
    st.markdown("""
    **Aim:** Observe the impact of the presence or absence of an eavesdropper (Eve) on key agreement and QBER in BB84 protocol.
    """)
    st.subheader("Procedure")
    st.markdown("""
    1. Select number of qubits (8–128, powers of 2).  
    2. Run BB84 simulation **with** or **without** Eve.  
    3. Observe the BB84 protocol process table.  
    4. Check Alice & Bob key agreement and QBER.  
    5. Compare QBER with and without Eve using the compare button.  
    """)

    # Controls
    num_qubits = st.sidebar.select_slider("Number of Qubits", options=[8,16,32,64,128], value=16)
    run_with_eve = st.sidebar.button("Run with Eve")
    run_without_eve = st.sidebar.button("Run without Eve")
    compare_qber = st.sidebar.button("Compare QBER")
    reset_expt1 = st.sidebar.button("Reset")

    if run_with_eve:
        df, alice_key, bob_key, qber_eve = bb84_simulation(num_qubits, eve=True)
        st.subheader("BB84 Protocol Process (With Eve)")
        st.dataframe(df)
        st.write(f"Final Alice Key (Kept Bits): {alice_key}")
        st.write(f"Final Bob Key (Kept Bits): {bob_key}")
        st.write(f"QBER: {qber_eve:.3f}")

    if run_without_eve:
        df, alice_key, bob_key, qber_no_eve = bb84_simulation(num_qubits, eve=False)
        st.subheader("BB84 Protocol Process (Without Eve)")
        st.dataframe(df)
        st.write(f"Final Alice Key (Kept Bits): {alice_key}")
        st.write(f"Final Bob Key (Kept Bits): {bob_key}")
        st.write(f"QBER: {qber_no_eve:.3f}")

    if compare_qber:
        # Run both
        _, _, _, qber_eve = bb84_simulation(num_qubits, eve=True)
        _, _, _, qber_no_eve = bb84_simulation(num_qubits, eve=False)
        st.subheader("QBER Comparison")
        fig, ax = plt.subplots()
        ax.bar(["With Eve","Without Eve"], [qber_eve,qber_no_eve], color=['red','green'])
        ax.set_ylabel("QBER")
        ax.set_title("QBER Comparison")
        st.pyplot(fig)

# --------------------------------------------------------
if experiment == "Experiment 2: BB84 by varying qubits":
    st.header("Experiment 2: Impact of Number of Qubits on QBER")
    st.markdown("""
    **Aim:** Observe how the number of qubits affects the Quantum Bit Error Rate (QBER) in BB84 protocol.
    """)
    st.subheader("Procedure")
    st.markdown("""
    1. Select number of qubits (8–128, powers of 2).  
    2. Run BB84 simulation **with** or **without** Eve.  
    3. Observe BB84 table for each qubit selection.  
    4. Check final key and QBER.  
    5. Plot number of qubits vs QBER.  
    """)

    num_qubits_list = [8,16,32,64,128]
    with_eve = st.sidebar.checkbox("Run With Eve", value=False)
    plot_qubits_vs_qber = st.sidebar.button("Plot Qubit vs QBER")
    reset_expt2 = st.sidebar.button("Reset")

    all_keys = []
    qber_list = []

    bb84_tables = []

    for nq in num_qubits_list:
        df, alice_key, bob_key, qber = bb84_simulation(nq, eve=with_eve)
        df["Number of Qubits"] = nq
        bb84_tables.append(df)
        qber_list.append(qber)
        all_keys.append(alice_key)

    # Combine tables
    combined_df = pd.concat(bb84_tables, ignore_index=True)
    st.subheader("BB84 Protocol Process for All Qubit Selections")
    st.dataframe(combined_df)

    qber_summary = pd.DataFrame({
        "Number of Qubits": num_qubits_list,
        "QBER": qber_list
    })
    st.subheader("Summary: Number of Qubits vs QBER")
    st.dataframe(qber_summary)

    if plot_qubits_vs_qber:
        fig, ax = plt.subplots()
        ax.plot(num_qubits_list, qber_list, marker='o', label=("With Eve" if with_eve else "Without Eve"))
        ax.set_xlabel("Number of Qubits")
        ax.set_ylabel("QBER")
        ax.set_title("Number of Qubits vs QBER")
        ax.legend()
        st.pyplot(fig)

# --------------------------------------------------------
if experiment == "Experiment 3: Impact of channel noise":
    st.header("Experiment 3: Impact of Channel Noise on QBER")
    st.markdown("""
    **Aim:** Observe how channel noise affects QBER in BB84 protocol.
    Noise is represented as the probability (0–1) that a transmitted qubit flips randomly.
    """)
    st.subheader("Procedure")
    st.markdown("""
    1. Select channel noise level (0–0.1).  
    2. Run BB84 simulation with selected noise.  
    3. Observe BB84 table and QBER.  
    4. Plot channel noise vs QBER.  
    """)

    noise_level = st.sidebar.slider("Channel Noise Probability", min_value=0.0, max_value=0.1, step=0.01, value=0.01)
    run_noise = st.sidebar.button("Run Simulation")
    plot_noise = st.sidebar.button("Plot Noise vs QBER")

    if run_noise:
        df, alice_key, bob_key, qber = bb84_simulation(32, eve=False, noise=noise_level)
        st.subheader(f"BB84 Protocol Process with Noise {noise_level}")
        st.dataframe(df)
        st.write(f"Alice Key: {alice_key}")
        st.write(f"Bob Key: {bob_key}")
        st.write(f"QBER: {qber:.3f}")

    if plot_noise:
        noise_vals = np.arange(0, 0.11, 0.01)
        qber_vals = []
        for n in noise_vals:
            _, _, _, q = bb84_simulation(32, eve=False, noise=n)
            qber_vals.append(q)
        fig, ax = plt.subplots()
        ax.plot(noise_vals, qber_vals, marker='o')
        ax.set_xlabel("Channel Noise Probability")
        ax.set_ylabel("QBER")
        ax.set_title("Channel Noise vs QBER")
        st.pyplot(fig)

# --------------------------------------------------------
if experiment == "Experiment 4: Impact of distance":
    st.header("Experiment 4: Impact of Distance between Alice & Bob on QBER")
    st.markdown("""
    **Aim:** Observe how increasing distance (or simulating it as a proxy for decoherence/noise) affects QBER.
    """)
    st.subheader("Procedure")
    st.markdown("""
    1. Select a distance value (arbitrary units).  
    2. Run BB84 simulation assuming higher distance increases error probability.  
    3. Observe BB84 table and QBER.  
    4. Plot distance vs QBER.  
    """)

    distance = st.sidebar.slider("Distance (arbitrary units)", min_value=1, max_value=20, step=1, value=5)
    run_distance = st.sidebar.button("Run Simulation")
    plot_distance = st.sidebar.button("Plot Distance vs QBER")

    if run_distance:
        # We simulate distance as noise factor proportional to distance
        noise_factor = distance * 0.01
        df, alice_key, bob_key, qber = bb84_simulation(32, eve=False, noise=noise_factor)
        st.subheader(f"BB84 Protocol Process at Distance {distance}")
        st.dataframe(df)
        st.write(f"Alice Key: {alice_key}")
        st.write(f"Bob Key: {bob_key}")
        st.write(f"QBER: {qber:.3f}")

    if plot_distance:
        distances = np.arange(1,21)
        qber_vals = []
        for d in distances:
            noise_factor = d*0.01
            _, _, _, q = bb84_simulation(32, eve=False, noise=noise_factor)
            qber_vals.append(q)
        fig, ax = plt.subplots()
        ax.plot(distances, qber_vals, marker='o')
        ax.set_xlabel("Distance (arbitrary units)")
        ax.set_ylabel("QBER")
        ax.set_title("Distance vs QBER")
        st.pyplot(fig)

    

   
