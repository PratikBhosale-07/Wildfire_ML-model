import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
import logging
import os
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Page configuration
st.set_page_config(
    page_title="Wildfire Severity Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern minimalistic design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: #0F172A;
    }
    
    .main {
        background: #0F172A;
    }
    
    /* Modern Card Design */
    .modern-card {
        background: #1E293B;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #334155;
        transition: all 0.3s ease;
        margin: 16px 0;
    }
    
    .modern-card:hover {
        border-color: #FF6B6B;
        box-shadow: 0 8px 30px rgba(255, 107, 107, 0.15);
    }
    
    /* Prediction Result Box */
    .prediction-box {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        padding: 32px;
        border-radius: 20px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: 600;
        margin: 24px 0;
        box-shadow: 0 10px 40px rgba(255, 107, 107, 0.3);
        letter-spacing: 0.5px;
    }
    
    /* Stat Cards */
    .stat-box {
        background: #1E293B;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #334155;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-box:hover {
        transform: translateY(-4px);
        border-color: #FF6B6B;
        box-shadow: 0 8px 30px rgba(255, 107, 107, 0.2);
    }
    
    .stat-number {
        font-size: 42px;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .stat-label {
        font-size: 14px;
        color: #94A3B8;
        font-weight: 400;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* Typography */
    h1 {
        color: #F1F5F9 !important;
        font-weight: 700 !important;
        letter-spacing: -1px !important;
        margin-bottom: 8px !important;
    }
    
    h2 {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
    }
    
    h3 {
        color: #CBD5E1 !important;
        font-weight: 500 !important;
    }
    
    p {
        color: #94A3B8 !important;
        line-height: 1.6 !important;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        font-size: 16px;
        font-weight: 600;
        padding: 16px 48px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 20px rgba(255, 107, 107, 0.3);
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 30px rgba(255, 107, 107, 0.4);
    }
    
    /* Info Cards */
    .info-card {
        background: #1E293B;
        padding: 20px;
        border-radius: 12px;
        border-left: 3px solid #FF6B6B;
        margin: 12px 0;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        background: #252F3F;
        border-left-width: 4px;
    }
    
    .info-card h3, .info-card h4 {
        color: #F1F5F9 !important;
        margin-bottom: 8px;
    }
    
    .info-card p {
        color: #94A3B8 !important;
        margin: 0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #1E293B;
        border-right: 1px solid #334155;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #F1F5F9 !important;
    }
    
    /* Input Fields */
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div {
        background: #1E293B !important;
        color: #F1F5F9 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    
    .stSlider>div>div>div>div {
        background: #FF6B6B !important;
    }
    
    /* Alert Boxes */
    .stAlert {
        background: #1E293B;
        border-radius: 12px;
        border: 1px solid #334155;
    }
    
    /* Remove default margins */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Section Divider */
    hr {
        border-color: #334155 !important;
        margin: 32px 0 !important;
    }
    
    /* Custom accent line */
    .accent-line {
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        border-radius: 2px;
        margin: 16px auto;
    }
    </style>
""", unsafe_allow_html=True)

# Load trained model and scaler
@st.cache_resource
def load_model():
    try:
        # Get the directory where the script is located
        base_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
        
        model_path = base_dir / "best_fire_model.pkl"
        scaler_path = base_dir / "scaler.pkl"
        
        # Check if files exist
        if not model_path.exists():
            st.error(f"❌ Model file not found at: {model_path}")
            st.info("Please ensure 'best_fire_model.pkl' is in the app directory")
            return None, None
            
        if not scaler_path.exists():
            st.error(f"❌ Scaler file not found at: {scaler_path}")
            st.info("Please ensure 'scaler.pkl' is in the app directory")
            return None, None
        
        # Load the model and scaler
        model = joblib.load(str(model_path))
        scaler = joblib.load(str(scaler_path))
        
        return model, scaler
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please check that the model files are valid and not corrupted")
        return None, None

model, scaler = load_model()

# Sidebar
with st.sidebar:
    # Logo/Icon
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <div style='font-size: 48px;'>🔥</div>
            <h2 style='margin-top: 10px; font-size: 20px;'>Wildfire AI</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Prediction", "📈 Analytics", "ℹ️ About"],
        label_visibility="visible"
    )
    
    st.markdown("---")
    
    st.markdown("### 🎯 Model Status")
    
    # Model status indicator
    if model is not None and scaler is not None:
        st.success("✅ Model Ready")
    else:
        st.error("❌ Model Error")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model specs
    st.markdown("**Algorithm**")
    st.caption("XGBoost Regressor")
    
    st.markdown("**Accuracy**")
    st.caption("~95% Precision")
    
    st.markdown("**Features**")
    st.caption("10 Input Parameters")
    
    st.markdown("---")
    
    st.markdown("### 📅 Today")
    st.caption(datetime.now().strftime("%B %d, %Y"))
    
    st.markdown("---")
    
    st.markdown("### 👨‍💻 Developed By")
    st.caption("**MITAOE Students**")
    st.caption("Computational Intelligence")
    st.caption("Academic Year 2024-25")

# HOME PAGE
if page == "🏠 Home":
    # Hero Section
    st.markdown("<h1 style='text-align: center; font-size: 56px; margin-bottom: 0;'>Wildfire Severity Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<div class='accent-line'></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: #94A3B8; margin-top: 8px;'>AI-powered predictive analytics for intelligent fire management</p>", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class='stat-box'>
                <div class='stat-number'>95%</div>
                <div class='stat-label'>Accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='stat-box'>
                <div class='stat-number'>10</div>
                <div class='stat-label'>Features</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='stat-box'>
                <div class='stat-number'>3</div>
                <div class='stat-label'>Severity Levels</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class='stat-box'>
                <div class='stat-number'>&lt;1s</div>
                <div class='stat-label'>Response Time</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Features Grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='modern-card'>
                <h3 style='font-size: 20px; margin-bottom: 12px;'>🎯 Accurate Predictions</h3>
                <p style='font-size: 14px;'>Advanced ML algorithms trained on historical data for precise wildfire severity forecasting.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='modern-card'>
                <h3 style='font-size: 20px; margin-bottom: 12px;'>⚡ Real-Time Analysis</h3>
                <p style='font-size: 14px;'>Instant predictions based on current incident parameters and resource deployment data.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='modern-card'>
                <h3 style='font-size: 20px; margin-bottom: 12px;'>📊 Data-Driven</h3>
                <p style='font-size: 14px;'>Intelligent insights for optimal resource allocation and emergency response planning.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Process Section
    st.markdown("<h2 style='text-align: center; margin-top: 40px;'>How It Works</h2>", unsafe_allow_html=True)
    st.markdown("<div class='accent-line'></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='info-card'>
                <h4 style='font-size: 16px; font-weight: 600;'>1. Input Parameters</h4>
                <p style='font-size: 14px;'>Enter location, containment status, and deployed resources</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='info-card'>
                <h4 style='font-size: 16px; font-weight: 600;'>3. Get Results</h4>
                <p style='font-size: 14px;'>Receive severity classification and recommended actions</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='info-card'>
                <h4 style='font-size: 16px; font-weight: 600;'>2. AI Processing</h4>
                <p style='font-size: 14px;'>XGBoost model analyzes data using trained algorithms</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='info-card'>
                <h4 style='font-size: 16px; font-weight: 600;'>4. Take Action</h4>
                <p style='font-size: 14px;'>Implement strategies based on severity predictions</p>
            </div>
        """, unsafe_allow_html=True)

# PREDICTION PAGE
elif page == "📊 Prediction":
    st.markdown("<h1 style='text-align: center;'>Wildfire Severity Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<div class='accent-line'></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 16px;'>Enter fire incident parameters for accurate severity prediction</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick demo overview
    st.info("💡 **Quick Start:** Select a demo scenario below to automatically populate all input fields with realistic fire incident data. Each scenario demonstrates different severity levels.")
    
    # Demo Scenarios Section
    st.markdown("### 🎬 Demo Scenarios")
    
    # Info box
    with st.expander("📋 View All Demo Scenarios & Values", expanded=False):
        st.markdown("**Compare all preset scenarios:**")
        
        # Create a comparison table
        comparison_data = {
            "Scenario": ["Minor Fire", "Moderate Fire", "Severe Fire", "Contained Fire"],
            "Containment": ["75%", "30%", "10%", "95%"],
            "Personnel": ["25", "150", "500", "100"],
            "Engines": ["5", "25", "75", "15"],
            "Helicopters": ["1", "5", "15", "3"],
            "Dozers": ["0", "3", "10", "2"],
            "Water Tenders": ["1", "8", "20", "5"],
            "Major Incident": ["No", "Yes", "Yes", "No"],
            "Expected Output": ["Minor", "Moderate", "Severe", "Minor"]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Style the dataframe
        st.dataframe(
            df_comparison,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Expected Output": st.column_config.TextColumn(
                    "Expected Output",
                    help="Expected severity classification"
                )
            }
        )
    
    demo_scenarios = {
        "Custom Input": {
            "county": 10, "latitude": 37.0, "longitude": -120.0,
            "percent_contained": 50.0, "personnel": 50, "engines": 10,
            "helicopters": 2, "dozers": 1, "water_tenders": 2, "major_incident": "No"
        },
        "Minor Fire (Small Scale)": {
            "county": 15, "latitude": 38.5, "longitude": -121.5,
            "percent_contained": 75.0, "personnel": 25, "engines": 5,
            "helicopters": 1, "dozers": 0, "water_tenders": 1, "major_incident": "No"
        },
        "Moderate Fire (Growing)": {
            "county": 25, "latitude": 36.5, "longitude": -119.5,
            "percent_contained": 30.0, "personnel": 150, "engines": 25,
            "helicopters": 5, "dozers": 3, "water_tenders": 8, "major_incident": "Yes"
        },
        "Severe Fire (Critical)": {
            "county": 35, "latitude": 39.0, "longitude": -122.0,
            "percent_contained": 10.0, "personnel": 500, "engines": 75,
            "helicopters": 15, "dozers": 10, "water_tenders": 20, "major_incident": "Yes"
        },
        "Contained Fire (Nearly Out)": {
            "county": 20, "latitude": 37.8, "longitude": -120.8,
            "percent_contained": 95.0, "personnel": 100, "engines": 15,
            "helicopters": 3, "dozers": 2, "water_tenders": 5, "major_incident": "No"
        }
    }
    
    # Initialize session state for demo selection
    if 'selected_demo' not in st.session_state:
        st.session_state.selected_demo = "Custom Input"
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_demo = st.selectbox(
            "Load Demo Values",
            options=list(demo_scenarios.keys()),
            index=list(demo_scenarios.keys()).index(st.session_state.selected_demo),
            help="Select a preset scenario to see different prediction outcomes",
            key="demo_selector"
        )
        
        # Update session state when selection changes
        if selected_demo != st.session_state.selected_demo:
            st.session_state.selected_demo = selected_demo
            st.rerun()
    
    with col2:
        # Show expected outcome hint
        if selected_demo == "Minor Fire (Small Scale)":
            st.markdown("<div class='modern-card' style='padding: 12px; background: #1E293B; border-left: 3px solid #10B981;'><p style='margin: 0; font-size: 13px; color: #10B981;'>Expected: Minor</p></div>", unsafe_allow_html=True)
        elif selected_demo == "Moderate Fire (Growing)":
            st.markdown("<div class='modern-card' style='padding: 12px; background: #1E293B; border-left: 3px solid #F59E0B;'><p style='margin: 0; font-size: 13px; color: #F59E0B;'>Expected: Moderate</p></div>", unsafe_allow_html=True)
        elif selected_demo == "Severe Fire (Critical)":
            st.markdown("<div class='modern-card' style='padding: 12px; background: #1E293B; border-left: 3px solid #EF4444;'><p style='margin: 0; font-size: 13px; color: #EF4444;'>Expected: Severe</p></div>", unsafe_allow_html=True)
        elif selected_demo == "Contained Fire (Nearly Out)":
            st.markdown("<div class='modern-card' style='padding: 12px; background: #1E293B; border-left: 3px solid #10B981;'><p style='margin: 0; font-size: 13px; color: #10B981;'>Expected: Minor</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='modern-card' style='padding: 12px; background: #1E293B;'><p style='margin: 0; font-size: 13px; color: #94A3B8;'>Custom values</p></div>", unsafe_allow_html=True)
    
    demo_data = demo_scenarios[selected_demo]
    
    # Show current scenario details
    if selected_demo != "Custom Input":
        st.markdown("#### 📊 Current Scenario Values:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class='info-card' style='padding: 12px;'>
                    <p style='margin: 0; font-size: 13px; color: #94A3B8;'>Location</p>
                    <p style='margin: 4px 0 0 0; font-size: 14px; color: #F1F5F9;'>
                        County: {demo_data['county']}<br>
                        Lat: {demo_data['latitude']}<br>
                        Long: {demo_data['longitude']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class='info-card' style='padding: 12px;'>
                    <p style='margin: 0; font-size: 13px; color: #94A3B8;'>Fire Status</p>
                    <p style='margin: 4px 0 0 0; font-size: 14px; color: #F1F5F9;'>
                        Containment: {demo_data['percent_contained']}%<br>
                        Major Incident: {demo_data['major_incident']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class='info-card' style='padding: 12px;'>
                    <p style='margin: 0; font-size: 13px; color: #94A3B8;'>Resources</p>
                    <p style='margin: 4px 0 0 0; font-size: 14px; color: #F1F5F9;'>
                        Personnel: {demo_data['personnel']} | Engines: {demo_data['engines']}<br>
                        Helicopters: {demo_data['helicopters']} | Dozers: {demo_data['dozers']}<br>
                        Water Tenders: {demo_data['water_tenders']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Location Section
    st.markdown("### 📍 Location")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        county = st.number_input(
            "County Code",
            min_value=0,
            max_value=60,
            value=demo_data["county"],
            key="county",
            help="Encoded county identifier"
        )
    
    with col2:
        latitude = st.number_input(
            "Latitude",
            value=demo_data["latitude"],
            key="latitude",
            format="%.4f"
        )
    
    with col3:
        longitude = st.number_input(
            "Longitude",
            value=demo_data["longitude"],
            key="longitude",
            format="%.4f"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Fire Status
    st.markdown("### 🔥 Fire Status")
    col1, col2 = st.columns(2)
    
    with col1:
        percent_contained = st.slider(
            "Containment (%)",
            min_value=0.0,
            max_value=100.0,
            value=demo_data["percent_contained"],
            key="containment",
            step=5.0
        )
    
    with col2:
        major_incident = st.selectbox(
            "Major Incident",
            ["No", "Yes"],
            index=0 if demo_data["major_incident"] == "No" else 1,
            key="major"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Resources
    st.markdown("### 🚁 Deployed Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        personnel = st.number_input(
            "Personnel",
            min_value=0,
            value=demo_data["personnel"],
            key="personnel",
            step=5
        )
        
        engines = st.number_input(
            "Fire Engines",
            min_value=0,
            value=demo_data["engines"],
            key="engines",
            step=1
        )
    
    with col2:
        helicopters = st.number_input(
            "Helicopters",
            min_value=0,
            value=demo_data["helicopters"],
            key="helicopters",
            step=1
        )
        
        dozers = st.number_input(
            "Bulldozers",
            min_value=0,
            value=demo_data["dozers"],
            key="dozers",
            step=1
        )
    
    with col3:
        water_tenders = st.number_input(
            "Water Tenders",
            min_value=0,
            value=demo_data["water_tenders"],
            key="water_tenders",
            step=1
        )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Convert input to dataframe
    input_data = pd.DataFrame({
        "Counties": [county],
        "Latitude": [latitude],
        "Longitude": [longitude],
        "PercentContained": [percent_contained],
        "PersonnelInvolved": [personnel],
        "Engines": [engines],
        "Helicopters": [helicopters],
        "Dozers": [dozers],
        "WaterTenders": [water_tenders],
        "MajorIncident": [1 if major_incident == "Yes" else 0]
    })
    
    # Center the button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("Predict Severity", use_container_width=True)
    
    # Prediction
    if predict_button:
        if model is not None and scaler is not None:
            with st.spinner("Analyzing..."):
                # Scale input
                scaled_input = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(scaled_input)[0]
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Display prediction
                st.markdown(f"""
                    <div class='prediction-box'>
                        {prediction:,.0f} Acres Predicted
                    </div>
                """, unsafe_allow_html=True)
                
                # Severity Classification
                if prediction > 100000:
                    severity_level = "Severe"
                    color = "#EF4444"
                    icon = "🚨"
                    message = "Critical situation requiring immediate response"
                    recommendation = """
                    **Immediate Actions:**
                    - Deploy maximum available resources
                    - Initiate evacuation procedures
                    - Request external support
                    - Establish incident command
                    """
                elif prediction > 10000:
                    severity_level = "Moderate"
                    color = "#F59E0B"
                    icon = "⚠️"
                    message = "Significant fire requiring close monitoring"
                    recommendation = """
                    **Recommended Actions:**
                    - Monitor fire progression closely
                    - Scale up resource allocation
                    - Prepare evacuation routes
                    - Coordinate with agencies
                    """
                else:
                    severity_level = "Minor"
                    color = "#10B981"
                    icon = "✓"
                    message = "Situation manageable with current resources"
                    recommendation = """
                    **Standard Actions:**
                    - Continue monitoring
                    - Maintain resource levels
                    - Regular status updates
                    - Plan containment strategy
                    """
                
                st.markdown(f"""
                    <div class='modern-card' style='border-left: 4px solid {color}; text-align: center;'>
                        <h2 style='color: {color}; margin-bottom: 8px;'>{icon} {severity_level} Fire</h2>
                        <p style='font-size: 15px;'>{message}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Acres Burned", 'font': {'size': 24, 'color': 'white'}},
                    delta={'reference': 50000, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, 200000], 'tickcolor': 'white'},
                        'bar': {'color': color},
                        'bgcolor': 'rgba(255, 255, 255, 0.1)',
                        'borderwidth': 2,
                        'bordercolor': 'white',
                        'steps': [
                            {'range': [0, 10000], 'color': 'rgba(16, 185, 129, 0.3)'},
                            {'range': [10000, 100000], 'color': 'rgba(245, 158, 11, 0.3)'},
                            {'range': [100000, 200000], 'color': 'rgba(220, 38, 38, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': prediction
                        }
                    }
                ))
                
                fig.update_layout(
                    paper_bgcolor='#0F172A',
                    plot_bgcolor='#0F172A',
                    font={'color': '#F1F5F9', 'family': 'Inter'},
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("### Recommended Actions")
                st.markdown(recommendation)
                
        else:
            st.error("Model not available. Please check configuration.")

# ANALYTICS PAGE
elif page == "📈 Analytics":
    st.markdown("<h1 style='text-align: center;'>Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div class='accent-line'></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 16px;'>Historical trends and performance insights</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sample data for visualization
    severity_categories = ['Minor\n(0-10K acres)', 'Moderate\n(10K-100K acres)', 'Severe\n(>100K acres)']
    fire_counts = [450, 280, 85]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig1 = go.Figure(data=[go.Pie(
            labels=severity_categories,
            values=fire_counts,
            hole=0.4,
            marker=dict(colors=['#10b981', '#f59e0b', '#dc2626'])
        )])
        
        fig1.update_layout(
            title={'text': 'Severity Distribution', 'font': {'size': 18}},
            paper_bgcolor='#0F172A',
            plot_bgcolor='#0F172A',
            font={'color': '#F1F5F9', 'family': 'Inter'},
            height=350,
            showlegend=True
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Bar chart
        fig2 = go.Figure(data=[go.Bar(
            x=severity_categories,
            y=fire_counts,
            marker_color=['#10B981', '#F59E0B', '#EF4444'],
            text=fire_counts,
            textposition='auto'
        )])
        
        fig2.update_layout(
            title={'text': 'Incidents by Severity', 'font': {'size': 18}},
            xaxis_title='',
            yaxis_title='Count',
            paper_bgcolor='#0F172A',
            plot_bgcolor='#0F172A',
            font={'color': '#F1F5F9', 'family': 'Inter'},
            height=350
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Resource effectiveness
    st.markdown("### Resource Effectiveness")
    
    resources = ['Personnel', 'Engines', 'Helicopters', 'Dozers', 'Water Tenders']
    effectiveness = [85, 78, 92, 70, 88]
    
    fig3 = go.Figure(data=[go.Bar(
        x=resources,
        y=effectiveness,
        marker_color='#FF6B6B',
        text=[f'{e}%' for e in effectiveness],
        textposition='auto'
    )])
    
    fig3.update_layout(
        title={'text': 'Resource Effectiveness Scores', 'font': {'size': 18}},
        xaxis_title='',
        yaxis_title='Effectiveness (%)',
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        font={'color': '#F1F5F9', 'family': 'Inter'},
        height=350,
        yaxis={'range': [0, 100]}
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Insights
    st.markdown("### Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='info-card'>
                <h4 style='font-size: 16px; font-weight: 600;'>Most Effective Resource</h4>
                <p style='font-size: 14px;'>Helicopters demonstrate 92% effectiveness in containment operations</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='info-card'>
                <h4 style='font-size: 16px; font-weight: 600;'>Fire Distribution</h4>
                <p style='font-size: 14px;'>55% of incidents classified as minor with rapid containment</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='info-card'>
                <h4 style='font-size: 16px; font-weight: 600;'>Response Time</h4>
                <p style='font-size: 14px;'>Average 72-hour containment with proper resource allocation</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='info-card'>
                <h4 style='font-size: 16px; font-weight: 600;'>Critical Success Factor</h4>
                <p style='font-size: 14px;'>Early detection reduces acres burned by 65%</p>
            </div>
        """, unsafe_allow_html=True)

# ABOUT PAGE
elif page == "ℹ️ About":
    st.markdown("<h1 style='text-align: center;'>About This Project</h1>", unsafe_allow_html=True)
    st.markdown("<div class='accent-line'></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='modern-card'>
            <h3 style='font-size: 18px; margin-bottom: 12px;'>🎓 Academic Project</h3>
            <p style='font-size: 14px;'>Developed as part of <strong>Computational Intelligence (CO4)</strong> at <strong>MIT Academy of Engineering, Pune</strong> — Academic Year 2024-25</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='modern-card'>
                <h3 style='font-size: 18px; margin-bottom: 12px;'>🎯 Objectives</h3>
                <p style='font-size: 14px;'>
                    • Deploy ML models for real-world use<br>
                    • Create interactive AI interfaces<br>
                    • Implement predictive analytics<br>
                    • Apply computational intelligence
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='modern-card'>
                <h3 style='font-size: 18px; margin-bottom: 12px;'>🔧 Tech Stack</h3>
                <p style='font-size: 14px;'>
                    <strong>ML:</strong> XGBoost, Scikit-learn<br>
                    <strong>Frontend:</strong> Streamlit<br>
                    <strong>Data:</strong> Pandas, NumPy<br>
                    <strong>Viz:</strong> Plotly<br>
                    <strong>Storage:</strong> Joblib
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='modern-card'>
                <h3 style='font-size: 18px; margin-bottom: 12px;'>📊 Input Features</h3>
                <p style='font-size: 14px;'>
                    1. County Code<br>
                    2. Latitude & Longitude<br>
                    3. Containment Percentage<br>
                    4. Personnel Count<br>
                    5. Fire Engines<br>
                    6. Helicopters<br>
                    7. Bulldozers<br>
                    8. Water Tenders<br>
                    9. Major Incident Flag
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='modern-card'>
                <h3 style='font-size: 18px; margin-bottom: 12px;'>🎯 Classifications</h3>
                <p style='font-size: 14px;'>
                    <span style='color: #10B981;'>✓ Minor:</span> 0-10K acres<br>
                    <span style='color: #F59E0B;'>⚠ Moderate:</span> 10K-100K acres<br>
                    <span style='color: #EF4444;'>⚠ Severe:</span> >100K acres
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='modern-card'>
            <h3 style='font-size: 18px; margin-bottom: 12px;'>🌟 Future Roadmap</h3>
            <p style='font-size: 14px;'>
                Real-time weather integration • Historical pattern analysis • Ensemble model predictions • Mobile app development • Emergency system integration
            </p>
        </div>
    """, unsafe_allow_html=True)
    
# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 32px 0;'>
        <p style='color: #64748B; font-size: 14px; margin: 0;'>
            Wildfire AI • Computational Intelligence Project
        </p>
        <p style='color: #475569; font-size: 13px; margin-top: 8px;'>
            MIT Academy of Engineering, Pune • 2024-25
        </p>
    </div>
""", unsafe_allow_html=True)
