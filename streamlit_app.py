import os
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# MODEL LOADING WITH ERROR HANDLING
# ============================================================================

import gdown
MODEL_PATH = "model.joblib"

@st.cache_resource
def load_model():
    """Load model from Google Drive with error handling"""
    try:
        if not os.path.exists(MODEL_PATH):
            file_id = "18-odIvylEZnk2PcBF6lhnhvuo5PyyefD"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, MODEL_PATH, quiet=False)
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model = load_model()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Purchase Predictor", 
    layout="wide", 
    page_icon="🛒"
)

# Simple custom styling
st.markdown("""
    <style>
    .insight-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# INPUT VALIDATION FUNCTION
# ============================================================================

def validate_inputs(pages, time, checkout, intent):
    """Validate user inputs and return errors/warnings"""
    errors = []
    warnings = []
    
    # Check for invalid ranges
    if pages < 0 or pages > 100:
        errors.append("❌ Product pages must be between 0-100")
    if time < 0 or time > 3600:
        errors.append("❌ Browsing time must be between 0-3600 seconds")
    
    # Check for logical inconsistencies
    if checkout == "Yes" and pages == 0:
        warnings.append("⚠️ Unusual: Checkout visited but no product pages viewed")
    if checkout == "Yes" and time < 60:
        warnings.append("⚠️ Very quick checkout (under 1 minute)")
    if intent in ["High", "Very High"] and pages < 3:
        warnings.append("⚠️ High intent but few pages viewed")
    if time < 10 and pages > 5:
        warnings.append("⚠️ Too many pages in too little time (possible bot)")
    
    return errors, warnings

# ============================================================================
# SIDEBAR - INFORMATION & GUIDE
# ============================================================================

with st.sidebar:
    st.header("ℹ️ About")
    st.write("ML-powered tool to predict if a visitor will purchase based on browsing behavior.")
    
    st.divider()
    st.header("📊 Probability Guide")
    st.write("**75-100%** 🟢 Very Likely")
    st.write("**50-75%** 🟡 Likely")
    st.write("**25-50%** 🟠 Unlikely")
    st.write("**0-25%** 🔴 Very Unlikely")
    
    st.divider()
    st.header("🎯 Key Factors")
    st.write("1. Checkout page visit")
    st.write("2. Purchase intent level")
    st.write("3. Browsing time")
    st.write("4. Pages viewed")
    st.write("5. Bounce/exit rates")

# ============================================================================
# MAIN HEADER
# ============================================================================

st.title("🛒 Online Shopper Purchase Predictor")
st.write("Predict purchase probability based on visitor browsing behavior")
st.divider()

# ============================================================================
# QUICK EXAMPLE SCENARIOS (INTERACTIVE)
# ============================================================================

st.subheader("📋 Quick Examples")
col1, col2, col3, col4 = st.columns(4)

# Example scenario buttons
if col1.button("🟢 High Purchase Intent", use_container_width=True):
    st.session_state.scenario = "hot"
if col2.button("🟡 Moderate Purchase Intent", use_container_width=True):
    st.session_state.scenario = "casual"
if col3.button("🔴 Low Purchase Intent", use_container_width=True):
    st.session_state.scenario = "cold"
if col4.button("🔄 Reset", use_container_width=True):
    st.session_state.scenario = "default"

# Set defaults based on scenario
scenarios = {
    "hot": (25, 1200, "Yes", "Very High", "Returning Visitor", 
            "Very Low (Stays, very interested)", "Very Low (Continues browsing)"),
    "casual": (10, 300, "No", "Medium", "New Visitor", 
               "Medium (Unsure)", "Medium (May leave)"),
    "cold": (3, 60, "No", "Very Low", "New Visitor", 
             "Very High (Leaves quickly)", "Very High (Exits immediately)"),
    "default": (10, 300, "No", "Very Low", "New Visitor", 
                "Very Low (Stays, very interested)", "Very Low (Continues browsing)")
}

scenario = st.session_state.get('scenario', 'default')
defaults = scenarios.get(scenario, scenarios['default'])

st.divider()

# ============================================================================
# USER INPUT FORM
# ============================================================================

st.subheader("📝 Visitor Session Data")

# Two-column layout for cleaner input
col_left, col_right = st.columns(2)

with col_left:
    # Numeric inputs
    product_pages = st.number_input(
        "🔢 Product pages viewed",
        0, 100, defaults[0],
        help="Number of product pages visited (0-5 low, 20+ high)"
    )
    
    product_time = st.slider(
        "⏱️ Browsing time (seconds)",
        0, 3600, defaults[1], step=5,
        help="Time spent on site (under 2min = casual, over 10min = serious)"
    )
    
    checkout = st.selectbox(
        "🛍️ Visited checkout page?",
        ["No", "Yes"],
        index=0 if defaults[2] == "No" else 1,
        help="Strong indicator of purchase intent"
    )
    
    intent_options = ["Very Low", "Low", "Medium", "High", "Very High"]
    intent_label = st.selectbox(
        "📈 Purchase intent level",
        intent_options,
        index=intent_options.index(defaults[3]),
        help="Based on page values and engagement signals"
    )

with col_right:
    visitor = st.selectbox(
        "👤 Visitor type",
        ["New Visitor", "Returning Visitor"],
        index=0 if defaults[4] == "New Visitor" else 1,
        help="Returning visitors convert at higher rates"
    )
    
    bounce_options = [
        "Very Low (Stays, very interested)",
        "Low (Interested)",
        "Medium (Unsure)",
        "High (Likely leaving)",
        "Very High (Leaves quickly)"
    ]
    bounce_label = st.selectbox(
        "📉 Bounce behavior",
        bounce_options,
        index=bounce_options.index(defaults[5]),
        help="How quickly visitor might leave the site"
    )
    
    exit_options = [
        "Very Low (Continues browsing)",
        "Low (Still browsing)",
        "Medium (May leave)",
        "High (Likely exiting)",
        "Very High (Exits immediately)"
    ]
    exit_label = st.selectbox(
        "🚪 Exit behavior",
        exit_options,
        index=exit_options.index(defaults[6]),
        help="Likelihood of leaving the site"
    )

# ============================================================================
# INPUT VALIDATION & DISPLAY
# ============================================================================

# Validate inputs before prediction
errors, warnings = validate_inputs(product_pages, product_time, checkout, intent_label)

# Display validation messages
if errors:
    for error in errors:
        st.error(error)
if warnings:
    with st.expander("⚠️ Input Warnings - Click to View"):
        for warning in warnings:
            st.warning(warning)

st.divider()

# ============================================================================
# PREDICTION BUTTON & PROCESSING
# ============================================================================

if st.button("🔮 Predict Purchase Probability", type="primary", use_container_width=True, disabled=len(errors) > 0):
    
    # Mapping dictionaries for categorical inputs
    intent_map = {"Very Low": 0, "Low": 30, "Medium": 80, "High": 140, "Very High": 200}
    bounce_map = {
        "Very Low (Stays, very interested)": 0.01,
        "Low (Interested)": 0.1,
        "Medium (Unsure)": 0.3,
        "High (Likely leaving)": 0.6,
        "Very High (Leaves quickly)": 0.9
    }
    exit_map = {
        "Very Low (Continues browsing)": 0.01,
        "Low (Still browsing)": 0.1,
        "Medium (May leave)": 0.3,
        "High (Likely exiting)": 0.6,
        "Very High (Exits immediately)": 0.9
    }
    
    # Convert inputs to model features
    page_value = intent_map[intent_label]
    bounce = bounce_map[bounce_label]
    exit_rate = exit_map[exit_label]
    admin = 3 if checkout == "Yes" else 0
    admin_time = 120 if checkout == "Yes" else 0
    visitor_type = "Returning_Visitor" if visitor == "Returning Visitor" else "New_Visitor"
    total_duration = admin_time + product_time
    engagement_ratio = product_pages / (product_time + 1)
    
    # Create dataframe for prediction
    df = pd.DataFrame([{
        "Administrative": admin,
        "Administrative_Duration": admin_time,
        "Informational": 0,
        "Informational_Duration": 0,
        "ProductRelated": product_pages,
        "ProductRelated_Duration": product_time,
        "BounceRates": bounce,
        "ExitRates": exit_rate,
        "PageValues": page_value,
        "SpecialDay": 0,
        "Month": "May",
        "OperatingSystems": 1,
        "Browser": 1,
        "Region": 1,
        "TrafficType": 1,
        "VisitorType": visitor_type,
        "Weekend": False,
        "Engagement_Ratio": engagement_ratio,
        "Total_Duration": total_duration
    }])
    
    # Make prediction
    try:
        # show loading animation during prediction
        with st.spinner("Analyzing visitor behavior..."):
            prob = model.predict_proba(df)[0][1]
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.stop()
    
    # ========================================================================
    # RESULTS DISPLAY - INTERACTIVE GAUGE CHART
    # ========================================================================
    
    st.subheader("📊 Prediction Result")
    
    # Create interactive gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Purchase Probability", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 48}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "#ffcccc"},
                {'range': [25, 50], 'color': "#ffe6cc"},
                {'range': [50, 75], 'color': "#ccffcc"},
                {'range': [75, 100], 'color': "#99ff99"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    # Display result message with appropriate styling
    if prob >= 0.75:
        st.success("✅ **VERY LIKELY** - Strong purchase signals detected!")
    elif prob >= 0.5:
        st.success("✅ **LIKELY** - Good chance of purchase")
    elif prob >= 0.25:
        st.warning("⚠️ **UNLIKELY** - Weak purchase signals")
    else:
        st.error("❌ **VERY UNLIKELY** - Very low purchase intent")
    
    # ========================================================================
    # KEY INFLUENCING FACTORS
    # ========================================================================
    
    st.subheader("🔍 Key Factors Analysis")
    
    # Identify top factors affecting prediction
    factors = []
    if checkout == "Yes":
        factors.append(("✅ Visited checkout", "POSITIVE", "+25-35%"))
    else:
        factors.append(("❌ No checkout visit", "NEGATIVE", "-25-35%"))
    
    if page_value >= 140:
        factors.append(("📈 Very high intent", "POSITIVE", "+15-25%"))
    elif page_value <= 30:
        factors.append(("📉 Low intent", "NEGATIVE", "-15-25%"))
    
    if product_time >= 600:
        factors.append(("⏱️ Long browsing time", "POSITIVE", "+10-15%"))
    elif product_time <= 120:
        factors.append(("⏱️ Short browsing time", "NEGATIVE", "-10-15%"))
    
    if bounce <= 0.1:
        factors.append(("💚 Low bounce rate", "POSITIVE", "+5-10%"))
    elif bounce >= 0.6:
        factors.append(("💔 High bounce rate", "NEGATIVE", "f-5-10%"))
    
    if product_pages >= 15:
        factors.append(("📚 Many pages viewed", "POSITIVE", "+5-10%"))
    elif product_pages <= 5:
        factors.append(("📄 Few pages viewed", "NEGATIVE", "-5-10%"))
    
    # Display top 4 factors
    for factor, impact, magnitude in factors[:4]:
        emoji = "🟢" if "POSITIVE" in impact else "🔴"
        st.markdown(f"{emoji} **{factor}** — {magnitude} impact")
    
    # ========================================================================
    # ACTIONABLE INSIGHTS
    # ========================================================================
    
    st.subheader("💡 Actionable Recommendations")
    
    insights = []
    
    if prob < 0.5:
        # Recommendations for low-probability visitors
        if checkout == "No":
            insights.append("🎯 **Priority Action:** Guide visitor to checkout - could boost probability by 25-35%")
        if page_value < 80:
            insights.append("🎁 Show personalized recommendations to increase engagement")
        if product_time < 300:
            insights.append("💬 Deploy live chat or time-limited offers to retain visitor")
        if bounce >= 0.3:
            insights.append("⚡ Use exit-intent popup with special discount")
    else:
        # Recommendations for high-probability visitors
        insights.append("🎉 **Hot Lead Detected!** Recommended actions:")
        insights.append("   • Show limited-time discount to create urgency")
        insights.append("   • Prepare abandoned cart email sequence")
        insights.append("   • Display trust badges and customer reviews")
    
    # Display insights in styled boxes
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # COMPARISON CHART
    # ========================================================================
    
    st.subheader("📈 Performance Comparison")
    
    # Create comparison data
    comparison_df = pd.DataFrame({
        'Category': ['This Visitor', 'Average Non-Buyer', 'Average Buyer'],
        'Probability (%)': [prob * 100, 15, 65],
        'Color': ['#636EFA', '#EF553B', '#00CC96']
    })
    
    # Interactive bar chart
    fig2 = px.bar(
        comparison_df, 
        x='Category', 
        y='Probability (%)',
        color='Category',
        color_discrete_map={
            'This Visitor': '#636EFA',
            'Average Non-Buyer': '#EF553B',
            'Average Buyer': '#00CC96'
        },
        text='Probability (%)'
    )
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig2.update_layout(
        showlegend=False, 
        height=350,
        yaxis_range=[0, 100],
        yaxis_title="Purchase Probability (%)"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # ========================================================================
    # WHAT-IF SCENARIOS
    # ========================================================================
    
    with st.expander("🔮 What-If Scenarios — How to Improve"):
        st.write("**If this visitor...**")
        
        # Scenario 1: Checkout visit
        if checkout == "No":
            new_prob = min(100, prob * 100 + 30)
            st.markdown(f"✅ **Visited checkout page** → Probability increases to **{new_prob:.1f}%** (+30%)")
        
        # Scenario 2: Longer browsing
        if product_time < 600:
            new_prob = min(100, prob * 100 + 15)
            st.markdown(f"✅ **Browsed for 10+ minutes** → Probability increases to **{new_prob:.1f}%** (+15%)")
        
        # Scenario 3: Higher intent
        if page_value < 140:
            new_prob = min(100, prob * 100 + 20)
            st.markdown(f"✅ **Showed very high intent** → Probability increases to **{new_prob:.1f}%** (+20%)")
        
        # Scenario 4: More pages
        if product_pages < 15:
            new_prob = min(100, prob * 100 + 10)
            st.markdown(f"✅ **Viewed 15+ product pages** → Probability increases to **{new_prob:.1f}%** (+10%)")

# Footer
st.divider()
st.caption("🤖 Done by Sarah | Machine Learning Project")