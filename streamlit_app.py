import streamlit as st
import pandas as pd
import joblib

# load trained model
model = joblib.load("model.joblib")

# page layout
st.set_page_config(page_title="Online Shopper Purchase Predictor", layout="wide")

# center content
_, center, _ = st.columns([1,2,1])

with center:

    st.title("🛒 Online Shopper Purchase Predictor")

    st.write("Predict if a visitor will likely make a purchase based on browsing behaviour.")

    st.subheader("Visitor Session Scenario")

    # pages viewed
    product_pages = st.number_input(
        "Product pages viewed",
        0, 100, 10,
        help="More pages = more interest. Example: 0–5 low, 20+ high interest"
    )

    # browsing time (max reduced, easier range)
    product_time = st.slider(
        "Browsing time (seconds)",
        0, 3600, 300, step=5,
        help="Short time = browsing only. Long time = serious buyer"
    )

    # checkout visit
    checkout = st.selectbox(
        "Visited checkout page?",
        ["No", "Yes"],
        help="Visitors who reach checkout are much more likely to purchase"
    )

    # purchase intent using words instead of raw numbers
    intent_label = st.selectbox(
        "Purchase intent level",
        ["Very Low", "Low", "Medium", "High", "Very High"],
        help="Higher level means visitor shows stronger buying signals"
    )

    intent_map = {
        "Very Low": 0,
        "Low": 30,
        "Medium": 80,
        "High": 140,
        "Very High": 200
    }

    page_value = intent_map[intent_label]

    # visitor type
    visitor = st.selectbox(
        "Visitor type",
        ["New Visitor", "Returning Visitor"],
        help="Returning visitors are more likely to buy"
    )

    # bounce rate using words
    bounce_label = st.selectbox(
        "Bounce behaviour",
        ["Very Low (Stays, very interested)",
         "Low (Interested)",
         "Medium (Unsure)",
         "High (Likely leaving)",
         "Very High (Leaves quickly)"],
        help="Bounce rate measures how quickly visitor leaves website"
    )

    bounce_map = {
        "Very Low (Stays, very interested)": 0.01,
        "Low (Interested)": 0.1,
        "Medium (Unsure)": 0.3,
        "High (Likely leaving)": 0.6,
        "Very High (Leaves quickly)": 0.9
    }

    bounce = bounce_map[bounce_label]

    # exit rate using words
    exit_label = st.selectbox(
        "Exit behaviour",
        ["Very Low (Continues browsing)",
         "Low (Still browsing)",
         "Medium (May leave)",
         "High (Likely exiting)",
         "Very High (Exits immediately)"],
        help="Exit rate measures how likely visitor exits website"
    )

    exit_map = {
        "Very Low (Continues browsing)": 0.01,
        "Low (Still browsing)": 0.1,
        "Medium (May leave)": 0.3,
        "High (Likely exiting)": 0.6,
        "Very High (Exits immediately)": 0.9
    }

    exit_rate = exit_map[exit_label]

    # predict button
    if st.button("Predict Purchase Probability"):

        # convert inputs for model
        admin = 3 if checkout == "Yes" else 0
        admin_time = 120 if checkout == "Yes" else 0
        visitor_type = "Returning_Visitor" if visitor == "Returning Visitor" else "New_Visitor"
        total_time = admin_time + product_time

        # dataframe for model
        df = pd.DataFrame([{
            "Administrative": admin,
            "Informational": 0,
            "ProductRelated": product_pages,
            "Administrative_Duration": admin_time,
            "Informational_Duration": 0,
            "ProductRelated_Duration": product_time,
            "PageValues": page_value,
            "BounceRates": bounce,
            "ExitRates": exit_rate,
            "Weekend": 0,
            "SpecialDay": 0,
            "Month": "May",
            "Region": 1,
            "TrafficType": 1,
            "Browser": 1,
            "OperatingSystems": 1,
            "VisitorType": visitor_type,
            "Engagement_Ratio": 0,
            "Total_Duration": total_time
        }])

        # model prediction
        prob = model.predict_proba(df)[0][1]

        st.subheader("Prediction Result")

        if prob >= 0.5:
            st.success("Visitor is LIKELY to make a purchase")
        else:
            st.error("Visitor is UNLIKELY to make a purchase")

        st.metric("Purchase Probability", f"{prob:.2%}")
