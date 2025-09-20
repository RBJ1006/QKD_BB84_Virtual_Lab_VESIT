# QKD Virtual Lab - BB84 Protocol

This is a Streamlit-based virtual lab to simulate the BB84 Quantum Key Distribution (QKD) protocol.

## Features
- Introduction to QKD and BB84
- Step-by-step BB84 simulation (with/without eavesdropper)
- Random bit/basis generation, photon transmission, measurement
- Basis reconciliation and final key generation
- Error detection (QBER)

## How to run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Cloud
1. Fork/clone this repo to your GitHub account.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Create a new app and point it to `streamlit_app.py` in this repo.
4. Share your public app link with students.
