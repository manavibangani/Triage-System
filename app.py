import streamlit as st
from groq import Groq
import os
import json

st.title("AI Patient Triage System")

api_key = os.environ.get("my_api_key")

if "patients" not in st.session_state:
    st.session_state.patients = []

def get_triage(vitals, api_key):
    client = Groq(api_key=api_key)

    prompt = f"""
    You are a medical triage AI in a hospital emergency department.

    Analyze these patient vitals and return ONLY valid JSON.
    Do not include markdown or extra text.

    Patient Details:
    - Name: {vitals['name']}
    - Age: {vitals['age']}
    - Chief Complaint: {vitals['complaint']}
    - Heart Rate: {vitals['heart_rate']} bpm
    - Systolic BP: {vitals['bp_systolic']} mmHg
    - Diastolic BP: {vitals['bp_diastolic']} mmHg
    - SpO2: {vitals['spo2']}%
    - Temperature: {vitals['temperature']} °F

    Return ONLY this JSON format:
    {{
      "priority": "P1/P2/P3/P4/P5",
      "wait_time": "Immediate / 15 mins / 30 mins / 1 hour / 2+ hours",
      "condition": "short condition label",
      "reasoning": "brief explanation"
    }}

    Priority guide:
    P1 = Immediate life threat
    P2 = Emergency
    P3 = Urgent
    P4 = Semi-urgent
    P5 = Non-urgent
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    clean = response.choices[0].message.content.strip()

    clean = clean.replace("```json", "").replace("```", "")

    return json.loads(clean)

name = st.text_input("Patient Name")

complaint = st.text_input(
    "Chief Complaint (e.g. chest pain, breathlessness)"
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=None,
    placeholder="Enter age"
)

heart_rate = st.number_input(
    "Heart Rate (bpm)",
    min_value=0,
    max_value=300,
    value=None,
    placeholder="Enter heart rate"
)

bp_systolic = st.number_input(
    "Systolic BP (mmHg)",
    min_value=0,
    max_value=300,
    value=None,
    placeholder="Enter systolic BP"
)

bp_diastolic = st.number_input(
    "Diastolic BP (mmHg)",
    min_value=0,
    max_value=200,
    value=None,
    placeholder="Enter diastolic BP"
)

spo2 = st.number_input(
    "Oxygen Saturation SpO2 (%)",
    min_value=0,
    max_value=100,
    value=None,
    placeholder="Enter SpO2"
)

temperature = st.number_input(
    "Temperature (°F)",
    min_value=94.0,
    max_value=108.0,
    value=None,
    placeholder="Enter temperature"
)

submitted = st.button("Add Patient")

if submitted:

    if not api_key:
        st.error("API key not found!")

    elif not name:
        st.error("Please enter a patient name!")

    else:

        vitals = {
            "name": name,
            "complaint": complaint,
            "age": age,
            "heart_rate": heart_rate,
            "bp_systolic": bp_systolic,
            "bp_diastolic": bp_diastolic,
            "spo2": spo2,
            "temperature": temperature
        }

        try:
            with st.spinner("Analyzing patient vitals..."):
                result = get_triage(vitals, api_key)

            st.session_state.patients.append({**vitals, **result})

            st.success(f"Patient {name} added successfully!")

        except Exception as e:
            st.error(f"Error: {e}")

st.divider()

st.subheader("Patient Queue")

priority_order = {
    "P1": 1,
    "P2": 2,
    "P3": 3,
    "P4": 4,
    "P5": 5
}

if len(st.session_state.patients) == 0:
    st.info("No patients added yet.")

else:

    sorted_patients = sorted(
        st.session_state.patients,
        key=lambda x: priority_order.get(x["priority"], 999)
    )

    for patient in sorted_patients:

        if patient["priority"] == "P1":
            st.error(
                f"{patient['priority']} | {patient['name']} | "
                f"{patient['condition']} | Wait: {patient['wait_time']}"
            )

        elif patient["priority"] == "P2":
            st.warning(
                f"{patient['priority']} | {patient['name']} | "
                f"{patient['condition']} | Wait: {patient['wait_time']}"
            )

        else:
            st.success(
                f"{patient['priority']} | {patient['name']} | "
                f"{patient['condition']} | Wait: {patient['wait_time']}"
            )

        st.write(patient["reasoning"])

        st.divider()

