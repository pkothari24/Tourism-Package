import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# 1. Download and load the latest travel model artifact from Hugging Face Hub
@st.cache_resource # Caches the model so it doesn't download on every user click
def load_production_model():
    model_path = hf_hub_download(
        repo_id="praneeth232/machine_failure_model", 
        filename="best_wellness_package_model_v1.joblib"
    )
    return joblib.load(model_path)

model = load_production_model()

# 2. Streamlit UI Design
st.set_page_config(page_title="Wellness Travel Insights", layout="wide")

st.title("🌴 Wellness Tourism Package Conversion Predictor")
st.write("""
This analytics application evaluates the conversion probability of potential clients for the luxury Wellness Tourism Package. 
Input the customer demographics and sales team interaction metrics below to evaluate purchasing likelihood.
""")

st.write("---")

# Organize inputs into columns for clean dashboard scannability
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Profile Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income (Gross)", min_value=0, max_value=500000, value=25000, step=500)
    city_tier = st.selectbox("City Tier (1 = Highest)", [1, 2, 3])

with col2:
    st.subheader("📞 Interaction & Group Dynamics")
    typeof_contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("Duration of Sales Pitch (mins)", min_value=0.0, max_value=200.0, value=15.0, step=0.5)
    num_followups = st.slider("Number of Post-Pitch Follow-ups", min_value=1, max_value=10, value=3)
    pitch_satisfaction = st.slider("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
    num_trips = st.number_input("Average Annual Trips Taken", min_value=0, max_value=20, value=2, step=1)

    st.write("**Additional Logistics:**")
    passport = st.checkbox("Holds Valid Passport", value=False)
    own_car = st.checkbox("Owns a Car", value=False)

    # Family variables
    num_persons = st.number_input("Total Persons Visiting", min_value=1, max_value=10, value=2)
    num_children = st.number_input("Number of Children Accompanying (<5 years)", min_value=0, max_value=5, value=0)
    preferred_stars = st.selectbox("Preferred Hotel Star Rating", [3, 4, 5])

st.write("---")

# 3. Assemble inputs exactly matching the training feature set dataframe structure
# Boolean fields are translated into binary integers (0/1) for pipeline processing compatibility
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': num_persons,
    'NumberOfFollowups': num_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_stars,
    'MaritalStatus': marital_status,
    'NumberOfTrips': num_trips,
    'Passport': 1 if passport else 0,
    'PitchSatisfactionScore': pitch_satisfaction,
    'OwnCar': 1 if own_car else 0,
    'NumberOfChildrenVisiting': num_children,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])

# 4. Scoring Engine Prediction execution 
if st.button("Analyze Purchase Likelihood", type="primary", use_container_width=True):
    try:
        # Run inference using the pipeline object
        prediction = model.predict(input_data)[0]

        # Calculate prediction probabilities if supported by the model pipeline
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0][1] * 100
            st.metric(label="Conversion Confidence Status", value=f"{prob:.1f}% Chance")

        st.subheader("Target Action Assessment Result:")
        if prediction == 1:
            st.success("🎯 **High Conversion Likelihood:** The customer is predicted to buy the Wellness Tourism Package.")
        else:
            st.warning("⚠️ **Low Conversion Likelihood:** The customer is unlikely to buy the package given current configurations.")

    except Exception as e:
        st.error(f"Inference execution failed. Check features mapping compatibility. Error: {str(e)}")
