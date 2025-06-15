import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
import joblib  # Added to fix NameError

st.set_page_config(page_title="Healthcare Disparities Dashboard", layout="wide", initial_sidebar_state="expanded")

# Dynamic path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of dashboard.py (src/)
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Root of the project
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")

@st.cache_data
def load_data(file_path, chunksize=10000):
    """Load CSV in chunks to handle large files efficiently."""
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    try:
        # Read CSV in chunks and concatenate
        chunks = pd.read_csv(file_path, chunksize=chunksize)
        df = pd.concat(chunks, ignore_index=True)
        return df
    except Exception as e:
        st.error(f"Error loading file {file_path}: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    return joblib.load(model_path)

def play_alert():
    try:
        with open(os.path.join(BASE_DIR, "alert.wav"), "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav")
    except FileNotFoundError:
        st.warning("Alert sound file not found.")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Disparity Metrics", "Risk Scores", "Interventions", "Audit Log", "Predict Risk", "SHAP Analysis", "Feedback", "Report Disparity", "Help"])

if page == "Home":
    st.title("Healthcare Disparities Dashboard")
    st.markdown("Monitor and address disparities in patient care with real-time insights.")
    patients = load_data(os.path.join(DATA_PATH, "preprocessed_data.csv"))
    if not patients.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(patients, names="GENDER", title="Patient Distribution by Gender", hole=0.3)
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(patients, names="RACE", title="Patient Distribution by Race", hole=0.3)
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
    complaints_file = os.path.join(RESULTS_PATH, "complaints.csv")
    if os.path.exists(complaints_file):
        complaints = load_data(complaints_file)
        complaint_counts = complaints['category'].value_counts()
        threshold = 2
        high_complaints = complaint_counts[complaint_counts > threshold].index.tolist()
        if high_complaints:
            st.error(f"ALERT: High complaints in {', '.join(high_complaints)}. Authorities notified.")
            play_alert()
        fig = px.bar(complaint_counts, title="Complaints by Category", labels={'value': 'Count', 'index': 'Category'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No complaints reported yet.")

elif page == "Disparity Metrics":
    st.title("Disparity Metrics")
    st.markdown("Compare treatment rates across demographic groups to identify disparities.")
    data = load_data(os.path.join(RESULTS_PATH, "fairness_metrics.csv"))
    sensitive_features = data['sensitive_feature'].unique()
    selected_sf = st.selectbox("Select Sensitive Feature", sensitive_features)
    groups = data[data['sensitive_feature'] == selected_sf][selected_sf].unique()
    selected_groups = st.multiselect("Select Groups", groups, default=groups)
    filtered_data = data[(data['sensitive_feature'] == selected_sf) & (data[selected_sf].isin(selected_groups))]
    if not filtered_data.empty:
        st.dataframe(filtered_data[[selected_sf, 'selection_rate']].rename(columns={selected_sf: 'Group', 'selection_rate': 'Selection Rate'}))
        fig = px.bar(filtered_data, x=selected_sf, y='selection_rate', title=f"Selection Rates by {selected_sf}", color=selected_sf,
                     labels={selected_sf: 'Group', 'selection_rate': 'Selection Rate'})
        st.plotly_chart(fig, use_container_width=True)
        if filtered_data['selection_rate'].max() - filtered_data['selection_rate'].min() > 0.1:
            st.warning(f"Significant disparity detected in {selected_sf}!")
            play_alert()
    else:
        st.warning("No data available for selected filters.")

elif page == "Risk Scores":
    st.title("Risk Scores")
    st.markdown("View the distribution of patient risk scores across demographic groups.")
    data = load_data(os.path.join(DATA_PATH, "preprocessed_data.csv")).sample(100, random_state=42)
    model = load_model(os.path.join(MODELS_PATH, "risk_model.pkl"))
    if model is not None:
        data["risk_score"] = model.predict_proba(data[['GENDER', 'RACE', 'ETHNICITY']])[:, 1]
        group_by = st.selectbox("Group By", ["GENDER", "RACE", "ETHNICITY"])
        fig = px.box(data, x=group_by, y="risk_score", title=f"Risk Score Distribution by {group_by}",
                     labels={group_by: group_by.capitalize(), 'risk_score': 'Risk Score'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("- **High scores** (near 1): Likely to receive treatment.\n- **Low scores** (near 0): Less likely to receive treatment.")
        st.download_button("Download Risk Scores", data.to_csv(index=False), "risk_scores.csv", "text/csv")
    else:
        st.error("Failed to load risk model. Please ensure risk_model.pkl exists in the models directory.")

elif page == "Interventions":
    st.title("Suggested Interventions")
    st.markdown("Review and implement tailored actions to address disparities.")
    interventions_file = os.path.join(RESULTS_PATH, "interventions.txt")
    if os.path.exists(interventions_file):
        with open(interventions_file, "r", encoding='utf-8') as f:
            intervention_text = f.read()
        st.text_area("Suggested Interventions", intervention_text, height=200)
        intervention_action = st.text_input("Enter custom intervention (optional)")
        if st.button("Implement Intervention"):
            action = intervention_action if intervention_action else intervention_text[:50] + '...'
            intervention = {'timestamp': datetime.now(), 'action': f'Implemented: {action}'}
            audit_log = pd.DataFrame([intervention])
            audit_file = os.path.join(RESULTS_PATH, "audit_log.csv")
            if os.path.exists(audit_file):
                audit_log.to_csv(audit_file, mode='a', header=False, index=False)
            else:
                audit_log.to_csv(audit_file, index=False)
            st.success("Intervention logged!")
            play_alert()
    else:
        st.error("Interventions file not found. Please run interventions.py to generate it.")

elif page == "Audit Log":
    st.title("Audit Log")
    st.markdown("Track all implemented interventions.")
    audit_file = os.path.join(RESULTS_PATH, "audit_log.csv")
    if os.path.exists(audit_file):
        data = load_data(audit_file)
        st.dataframe(data, use_container_width=True)
        st.download_button("Download Audit Report", data.to_csv(index=False), "audit_log.csv", "text/csv")
    else:
        st.info("No audit logs available yet.")

elif page == "Predict Risk":
    st.title("Predict Patient Risk")
    st.markdown("Calculate a patient's treatment likelihood based on demographics.")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", [0, 1, 2], format_func=lambda x: {0: "Male", 1: "Female", 2: "Unknown"}[x])
        race = st.selectbox("Race", [0, 1, 2, 3, 4, -1], format_func=lambda x: {0: "White", 1: "Black", 2: "Asian", 3: "Native", 4: "Other", -1: "Unknown"}[x])
    with col2:
        ethnicity = st.selectbox("Ethnicity", [0, 1, -1], format_func=lambda x: {0: "Non-Hispanic", 1: "Hispanic", -1: "Unknown"}[x])
    if st.button("Predict"):
        model = load_model(os.path.join(MODELS_PATH, "risk_model.pkl"))
        if model is not None:
            input_data = pd.DataFrame([[gender, race, ethnicity]], columns=["GENDER", "RACE", "ETHNICITY"])
            risk_score = model.predict_proba(input_data)[:, 1][0]
            st.success(f"Predicted Risk Score: {risk_score:.2f}")
            st.markdown("""
            ### How the Risk Score is Calculated
            - **Model**: RandomForestClassifier
            - **Inputs**: Gender, Race, Ethnicity
            - **Output**: Probability (0 to 1) of receiving treatment
            The model analyzes demographic patterns to predict treatment likelihood.
            """)
            if risk_score > 0.5:
                st.warning("High risk detected!")
                st.markdown("""
                ### Mitigation Strategies
                - Ensure equitable access to care.
                - Review treatment protocols for potential bias.
                - Provide additional resources if needed.
                """)
                play_alert()
            else:
                st.markdown("""
                ### Mitigation Strategies
                - Monitor for potential disparities.
                - Ensure follow-up care is offered.
                """)
        else:
            st.error("Failed to load risk model. Please ensure risk_model.pkl exists in the models directory.")

elif page == "SHAP Analysis":
    st.title("SHAP Analysis")
    st.markdown("Understand how demographic features influence treatment predictions.")
    shap_image = os.path.join(RESULTS_PATH, "shap_plot.png")
    if os.path.exists(shap_image):
        st.image(shap_image, caption="Feature Impact on Predictions")
        st.markdown("""
        ### What is SHAP Analysis?
        SHAP (SHapley Additive exPlanations) shows how each feature (e.g., Gender, Race) affects the model's prediction:
        - **Red bars**: Features increasing the likelihood of treatment.
        - **Blue bars**: Features decreasing the likelihood.
        - **Bar width**: Indicates the strength of the impact.
        This helps identify which demographics most influence treatment decisions.
        """)
    else:
        st.error("SHAP plot not found. Please run shap_analysis.py to generate it.")

elif page == "Feedback":
    st.title("Provide Feedback")
    st.markdown("Share your thoughts to improve the system.")
    feedback = st.text_area("Enter your feedback", height=150)
    if st.button("Submit Feedback"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feedback_entry = pd.DataFrame([[timestamp, feedback]], columns=['timestamp', 'feedback'])
        feedback_file = os.path.join(RESULTS_PATH, "feedback.csv")
        if os.path.exists(feedback_file):
            feedback_entry.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            feedback_entry.to_csv(feedback_file, index=False)
        st.success("Feedback submitted!")
        play_alert()
    feedback_file = os.path.join(RESULTS_PATH, "feedback.csv")
    if os.path.exists(feedback_file):
        st.subheader("Previous Feedback")
        st.dataframe(load_data(feedback_file), use_container_width=True)

elif page == "Report Disparity":
    st.title("Report a Disparity")
    st.markdown("Log any observed disparities for review.")
    categories = ["Delayed Treatment", "Incorrect Diagnosis", "Resource Allocation", "Staff Bias", "Other"]
    category = st.selectbox("Select Category", categories)
    description = st.text_area("Describe the disparity", height=150)
    department = st.text_input("Department/Location (optional)")
    complaints_file = os.path.join(RESULTS_PATH, "complaints.csv")
    if st.button("Submit Report"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        complaint_entry = pd.DataFrame([[timestamp, category, description, department]],
                                     columns=['timestamp', 'category', 'description', 'department'])
        if os.path.exists(complaints_file):
            complaint_entry.to_csv(complaints_file, mode='a', header=False, index=False)
        else:
            complaint_entry.to_csv(complaints_file, index=False)
        st.success("Report submitted!")
        play_alert()
    if os.path.exists(complaints_file):
        st.subheader("Previous Reports")
        st.dataframe(load_data(complaints_file), use_container_width=True)

elif page == "Help":
    st.title("User Guide")
    st.markdown("""
    ### Welcome!
    This dashboard helps you detect and address healthcare disparities. Here's how to navigate it:

    - **Home**: See patient demographics and complaint alerts.
    - **Disparity Metrics**: Compare treatment rates by group (e.g., Gender, Race).
    - **Risk Scores**: Check risk score distributions for patients.
    - **Interventions**: View and apply disparity solutions.
    - **Audit Log**: Review implemented actions.
    - **Predict Risk**: Assess treatment likelihood for a new patient.
    - **SHAP Analysis**: Learn what drives predictions.
    - **Feedback**: Share suggestions.
    - **Report Disparity**: Log issues you notice.

    ### Tips
    - Use the sidebar to switch pages.
    - Look for explanations under each section.
    - Contact support for help!

    ### Debug Information
    - **Current working directory**: {}
    - **Data directory contents**: {}
    """.format(os.getcwd(), os.listdir(DATA_PATH) if os.path.exists(DATA_PATH) else "Data directory not found"))