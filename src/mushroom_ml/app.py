import streamlit as st
import requests

st.title("Mushroom Edibility Predictor")

# Define possible options for each mushroom attribute
FEATURE_OPTIONS = {
    "cap_shape": ['b','c','x','f','k','s'],
    "cap_surface": ['f','g','y','s'],
    "cap_color": ['n','b','c','g','r','p','u','e','w','y'],
    "bruises": ['t','f'],
    "odor": ['a','l','c','y','f','m','n','p','s'],
    "gill_attachment": ['a','d','f','n'],
    "gill_spacing": ['c','w','d'],
    "gill_size": ['b','n'],
    "gill_color": ['k','n','b','h','g','r','o','p','u','e','w','y'],
    "stalk_shape": ['e','t'],
    "stalk_root": ['b','c','u','e','z','r','?'],
    "stalk_surface_above_ring": ['f','y','k','s'],
    "stalk_surface_below_ring": ['f','y','k','s'],
    "stalk_color_above_ring": ['n','b','c','g','o','p','e','w','y'],
    "stalk_color_below_ring": ['n','b','c','g','o','p','e','w','y'],
    "veil_type": ['p','u'],
    "veil_color": ['n','o','w','y'],
    "ring_number": ['n','o','t'],
    "ring_type": ['c','e','f','l','n','p','s','z'],
    "spore_print_color": ['k','n','b','h','r','o','u','w','y'],
    "population": ['a','c','n','s','v','y'],
    "habitat": ['g','l','m','p','u','w','d']
}

# Build input form in a grid layout
with st.form("mushroom_form"):
    cols = st.columns(4)
    selections = {}
    # Create a selectbox for each feature
    for idx, (feature, options) in enumerate(FEATURE_OPTIONS.items()):
        col = cols[idx % 4]
        label = feature.replace('_', ' ').title()
        selections[feature] = col.selectbox(label, options)
    submitted = st.form_submit_button("Predict")

# On submit, call the prediction API and display result
if submitted:
    feature_list = [selections[f] for f in FEATURE_OPTIONS]
    response = requests.post('http://localhost:8000/predict', json=feature_list, timeout=5)

    if response.ok:
        pred = response.json().get("prediction")
        if pred == 'e':
            st.success("🍄‍🟫 Edible", icon="✅")
        else:
            st.error("🍄 Poisonous", icon="❌")
    else:
        st.error(f"API error {response.status_code}: {response.text}")
