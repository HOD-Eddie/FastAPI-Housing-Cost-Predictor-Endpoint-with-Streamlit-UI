import streamlit as st
import requests

# 1. Configure the page headers
st.set_page_config(page_title="AI Real Estate Platform", layout="centered")
st.title("🏡 Smart Real Estate Listing Platform")
st.markdown("Adjust the block and structural parameters below to calculate a market valuation and write an AI advertisement script.")

# 2. Organize input controls into clean columns for a tight layout
col1, col2 = st.columns(2)

with col1:
    median_income = st.slider("Neighborhood Median Income ($10k scales)", min_value=0.5, max_value=15.0, value=8.3, step=0.1, help="e.g., 8.3 = $83,000 average income")
    house_age = st.slider("Median House Age (Years)", min_value=1.0, max_value=52.0, value=41.0, step=1.0)
    avg_rooms = st.slider("Average Rooms per Household", min_value=1.0, max_value=10.0, value=6.9, step=0.1)
    avg_bedrooms = st.slider("Average Bedrooms per Household", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

with col2:
    population = st.number_input("Block Group Total Population", min_value=5, max_value=35000, value=322, step=10)
    avg_occupancy = st.slider("Average Household Occupants", min_value=1.0, max_value=6.0, value=2.5, step=0.1)
    latitude = st.number_input("Geographical Latitude", min_value=32.0, max_value=42.0, value=37.88, format="%.2f")
    longitude = st.number_input("Geographical Longitude", min_value=-124.0, max_value=-114.0, value=-122.23, format="%.2f")

# 3. Add an interactive dropdown select box for our Station 2 LLM tone selector
marketing_tone = st.selectbox(
    "Select Advertising Copywriting Tone",
    ["luxury", "enthusiastic", "professional", "rustic", "cozy"]
)

st.markdown("---")

# 4. Process execution on button trigger click
if st.button("Generate Smart Evaluation & Listing", type="primary"):
    
    # Pack the exact JSON keys required by app/schemas/housing_schema.py
    payload = {
        "median_income": median_income,
        "house_age": house_age,
        "avg_rooms": avg_rooms,
        "avg_bedrooms": avg_bedrooms,
        "population": float(population), # Cast numeric field securely to float
        "avg_occupancy": avg_occupancy,
        "latitude": latitude,
        "longitude": longitude,
        "marketing_tone": marketing_tone
    }
    
    API_URL = "http://localhost:8000/api/generate-listing"
    
    try:
        with st.spinner("Processing multi-stage pipeline (Running Regressor + LLM)..."):
            response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # 5. Render results cleanly to the dashboard screen
            price = result["estimated_price_usd"]
            description = result["generated_description"]
            
            st.success("### 📊 Pipeline Results Analysis")
            
            # Display price formatted as currency
            st.metric(label="Calculated Fair Market Valuation", value=f"${price:,.2f}")
            
            st.write("#### 📝 AI Generated Marketing Description")
            st.info(description)
            
            st.caption(f"Pipeline Engine Version: `{result['model_version']}`")
            
        else:
            st.error(f"Validation Error (HTTP {response.status_code}): Could not process data parameters.")
            st.json(response.json()) # Outputs the Pydantic error detail box visually if keys fail
            
    except requests.exceptions.ConnectionError:
        st.error("Connection Refused. Please make sure your FastAPI application is running on port 8000!")
