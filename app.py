import json
import os
import re
import threading
from datetime import datetime

import streamlit as st

try:
    from groq import Groq
except ImportError:
    Groq = None


st.set_page_config(
    page_title="AI Patient Triage System",
    page_icon=":hospital:",
    layout="wide",
)


PRIORITY_ORDER = {"P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5}
WAIT_TIMES = {
    "P1": "Immediate",
    "P2": "15 mins",
    "P3": "30 mins",
    "P4": "1 hour",
    "P5": "2+ hours",
}
PRIORITY_LABELS = {
    "P1": "Immediate life threat",
    "P2": "Emergency",
    "P3": "Urgent",
    "P4": "Semi-urgent",
    "P5": "Non-urgent",
}


def read_secret(name, fallback=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, fallback)


def load_dotenv_values(path=".env"):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@st.cache_resource
def queue_store():
    return {"patients": [], "next_ticket": 1, "lock": threading.Lock()}


def priority_badge(priority):
    return f"{priority} | Score {PRIORITY_ORDER.get(priority, 5)}/5"


def clean_priority(value):
    match = re.search(r"P[1-5]", str(value or "").upper())
    return match.group(0) if match else "P5"


def rule_based_triage(vitals):
    complaint = vitals["complaint"].lower()
    age = vitals["age"]
    hr = vitals["heart_rate"]
    sbp = vitals["bp_systolic"]
    dbp = vitals["bp_diastolic"]
    spo2 = vitals["spo2"]
    temp = vitals["temperature"]

    critical = []
    emergency = []
    urgent = []

    if spo2 < 90:
        critical.append("oxygen saturation below 90%")
    elif spo2 < 94:
        emergency.append("low oxygen saturation")

    if sbp < 90:
        critical.append("very low systolic blood pressure")
    elif sbp < 100 or sbp >= 220 or dbp >= 120:
        emergency.append("dangerous blood pressure reading")

    if hr <= 40 or hr >= 150:
        critical.append("dangerous heart rate")
    elif hr <= 50 or hr >= 130:
        emergency.append("abnormal heart rate")
    elif hr < 60 or hr > 110:
        urgent.append("heart rate outside the usual range")

    if temp >= 105:
        critical.append("very high fever")
    elif temp >= 103 or temp <= 95:
        emergency.append("concerning temperature")
    elif temp >= 100.4:
        urgent.append("fever")

    high_risk_terms = [
        "chest pain",
        "stroke",
        "seizure",
        "unconscious",
        "faint",
        "severe bleeding",
        "breathlessness",
        "shortness of breath",
        "suicidal",
        "anaphylaxis",
    ]
    moderate_terms = [
        "fracture",
        "abdominal pain",
        "vomiting",
        "dehydration",
        "infection",
        "head injury",
        "burn",
    ]

    if any(term in complaint for term in high_risk_terms):
        emergency.append("high-risk chief complaint")
    elif any(term in complaint for term in moderate_terms):
        urgent.append("complaint may need timely evaluation")

    if age >= 75 and (emergency or urgent):
        emergency.append("older adult with concerning symptoms")

    if critical:
        priority = "P1"
        drivers = critical
        condition = "Critical vitals"
    elif emergency:
        priority = "P2"
        drivers = emergency
        condition = "High-risk presentation"
    elif urgent:
        priority = "P3"
        drivers = urgent
        condition = "Needs urgent review"
    elif complaint:
        priority = "P4"
        drivers = ["stable vitals with a reported complaint"]
        condition = "Stable presentation"
    else:
        priority = "P5"
        drivers = ["no concerning complaint or vital sign entered"]
        condition = "Low acuity"

    return {
        "priority": priority,
        "score": PRIORITY_ORDER[priority],
        "wait_time": WAIT_TIMES[priority],
        "condition": condition,
        "reasoning": "; ".join(dict.fromkeys(drivers)).capitalize() + ".",
        "source": "Local rules",
    }


def get_ai_triage(vitals, api_key):
    if Groq is None:
        raise RuntimeError("The groq package is not installed.")

    client = Groq(api_key=api_key)
    prompt = f"""
You are a medical triage decision-support assistant.

Analyze these patient details and return ONLY valid JSON. No markdown.

Patient Details:
- Age: {vitals['age']}
- Chief Complaint: {vitals['complaint']}
- Heart Rate: {vitals['heart_rate']} bpm
- Systolic BP: {vitals['bp_systolic']} mmHg
- Diastolic BP: {vitals['bp_diastolic']} mmHg
- SpO2: {vitals['spo2']}%
- Temperature: {vitals['temperature']} F

Return this JSON shape:
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
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    clean = response.choices[0].message.content.strip()
    clean = clean.replace("```json", "").replace("```", "").strip()
    result = json.loads(clean)
    priority = clean_priority(result.get("priority"))

    return {
        "priority": priority,
        "score": PRIORITY_ORDER[priority],
        "wait_time": result.get("wait_time") or WAIT_TIMES[priority],
        "condition": result.get("condition") or PRIORITY_LABELS[priority],
        "reasoning": result.get("reasoning") or "AI returned no explanation.",
        "source": "Groq AI",
    }


def run_triage(vitals, api_key):
    rules_result = rule_based_triage(vitals)

    if not api_key:
        return rules_result

    try:
        ai_result = get_ai_triage(vitals, api_key)
    except Exception as exc:
        rules_result["source"] = f"Local rules; AI unavailable ({exc})"
        return rules_result

    rule_rank = PRIORITY_ORDER[rules_result["priority"]]
    ai_rank = PRIORITY_ORDER[ai_result["priority"]]

    if rule_rank < ai_rank:
        rules_result["source"] = "Local rules override"
        rules_result["reasoning"] += " AI suggested a lower priority, so the safer local score was kept."
        return rules_result

    return ai_result


def sorted_patients(patients):
    return sorted(
        patients,
        key=lambda patient: (
            PRIORITY_ORDER.get(patient["priority"], 999),
            patient["created_at"],
        ),
    )


def show_patient_result(patient):
    priority = patient["priority"]
    banner = {
        "P1": st.error,
        "P2": st.warning,
        "P3": st.info,
        "P4": st.success,
        "P5": st.success,
    }.get(priority, st.info)

    banner(
        f"Ticket {patient['ticket']} | {priority_badge(priority)} | "
        f"{patient['condition']} | Estimated wait: {patient['wait_time']}"
    )
    st.caption(f"Assessment source: {patient['source']}")
    st.write(patient["reasoning"])


def require_staff_login(staff_pin):
    if not staff_pin:
        st.error("Staff queue is locked. Set TRIAGE_STAFF_PIN in your environment or Streamlit secrets.")
        return False

    if st.session_state.get("staff_authenticated"):
        return True

    with st.form("staff_login"):
        entered_pin = st.text_input("Staff PIN", type="password")
        submitted = st.form_submit_button("Unlock staff queue")

    if submitted and entered_pin == staff_pin:
        st.session_state.staff_authenticated = True
        st.rerun()
    elif submitted:
        st.error("Incorrect PIN.")

    return False


load_dotenv_values()
api_key = read_secret("my_api_key") or read_secret("GROQ_API_KEY")
staff_pin = read_secret("TRIAGE_STAFF_PIN")
store = queue_store()

st.title("AI Patient Triage System")
st.caption(
    "Decision-support prototype for hospital intake. Final triage decisions should be made by qualified clinical staff."
)

patient_tab, staff_tab = st.tabs(["Patient check-in", "Staff queue"])

with patient_tab:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("Patient intake")
        with st.form("patient_form", clear_on_submit=True):
            name = st.text_input("Patient Name")
            complaint = st.text_area(
                "Chief Complaint",
                placeholder="e.g. chest pain, breathlessness, fever",
                height=90,
            )

            col_a, col_b = st.columns(2)
            with col_a:
                age = st.number_input("Age", min_value=1, max_value=120, value=None)
                heart_rate = st.number_input("Heart Rate (bpm)", min_value=0, max_value=300, value=None)
                spo2 = st.number_input("Oxygen Saturation SpO2 (%)", min_value=0, max_value=100, value=None)
            with col_b:
                bp_systolic = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=300, value=None)
                bp_diastolic = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=200, value=None)
                temperature = st.number_input("Temperature (F)", min_value=94.0, max_value=108.0, value=None)

            submitted = st.form_submit_button("Add patient", use_container_width=True)

        if submitted:
            missing = []
            if not name.strip():
                missing.append("patient name")
            if not complaint.strip():
                missing.append("chief complaint")
            for label, value in {
                "age": age,
                "heart rate": heart_rate,
                "systolic BP": bp_systolic,
                "diastolic BP": bp_diastolic,
                "SpO2": spo2,
                "temperature": temperature,
            }.items():
                if value is None:
                    missing.append(label)

            if missing:
                st.error("Please enter: " + ", ".join(missing) + ".")
            else:
                vitals = {
                    "name": name.strip(),
                    "complaint": complaint.strip(),
                    "age": int(age),
                    "heart_rate": int(heart_rate),
                    "bp_systolic": int(bp_systolic),
                    "bp_diastolic": int(bp_diastolic),
                    "spo2": int(spo2),
                    "temperature": float(temperature),
                }

                with st.spinner("Assessing priority..."):
                    result = run_triage(vitals, api_key)

                with store["lock"]:
                    ticket = store["next_ticket"]
                    store["next_ticket"] += 1
                    patient = {
                        **vitals,
                        **result,
                        "ticket": ticket,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "status": "Waiting",
                    }
                    store["patients"].append(patient)

                st.session_state.latest_ticket = ticket
                st.success(f"Patient added. Ticket number: {ticket}")

    with right:
        st.subheader("Your latest result")
        latest_ticket = st.session_state.get("latest_ticket")
        latest_patient = next(
            (patient for patient in store["patients"] if patient["ticket"] == latest_ticket),
            None,
        )

        if latest_patient:
            show_patient_result(latest_patient)
        else:
            st.info("After check-in, this panel shows only your own ticket and triage result.")

        st.metric("Patients currently waiting", len([p for p in store["patients"] if p["status"] == "Waiting"]))

with staff_tab:
    st.subheader("Private staff queue")

    if require_staff_login(staff_pin):
        patients = sorted_patients(store["patients"])

        metric_cols = st.columns(5)
        for index, priority in enumerate(["P1", "P2", "P3", "P4", "P5"]):
            with metric_cols[index]:
                st.metric(
                    priority,
                    len([patient for patient in patients if patient["priority"] == priority and patient["status"] == "Waiting"]),
                )
                st.caption(PRIORITY_LABELS[priority])

        st.divider()

        if not patients:
            st.info("No patients added yet.")
        else:
            queue_rows = [
                {
                    "Ticket": patient["ticket"],
                    "Priority": priority_badge(patient["priority"]),
                    "Name": patient["name"],
                    "Age": patient["age"],
                    "Complaint": patient["complaint"],
                    "Condition": patient["condition"],
                    "Wait": patient["wait_time"],
                    "Status": patient["status"],
                    "Added": patient["created_at"],
                }
                for patient in patients
            ]
            st.dataframe(queue_rows, hide_index=True, use_container_width=True)

            st.subheader("Patient details")
            ticket_options = [patient["ticket"] for patient in patients]
            selected_ticket = st.selectbox("Select ticket", ticket_options)
            selected_patient = next(patient for patient in patients if patient["ticket"] == selected_ticket)

            show_patient_result(selected_patient)

            detail_cols = st.columns(4)
            detail_cols[0].metric("Heart rate", f"{selected_patient['heart_rate']} bpm")
            detail_cols[1].metric("Blood pressure", f"{selected_patient['bp_systolic']}/{selected_patient['bp_diastolic']}")
            detail_cols[2].metric("SpO2", f"{selected_patient['spo2']}%")
            detail_cols[3].metric("Temperature", f"{selected_patient['temperature']} F")

            with st.form("update_patient"):
                new_status = st.selectbox(
                    "Status",
                    ["Waiting", "In treatment", "Discharged"],
                    index=["Waiting", "In treatment", "Discharged"].index(selected_patient["status"]),
                )
                saved = st.form_submit_button("Update status")

            if saved:
                with store["lock"]:
                    selected_patient["status"] = new_status
                st.success("Status updated.")
                st.rerun()

        with st.expander("Queue controls"):
            if st.button("Clear discharged patients"):
                with store["lock"]:
                    store["patients"] = [
                        patient for patient in store["patients"] if patient["status"] != "Discharged"
                    ]
                st.success("Discharged patients cleared.")
                st.rerun()
