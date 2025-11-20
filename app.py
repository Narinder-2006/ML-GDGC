import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Page Config
st.set_page_config(
    page_title="Student Relation Status Predictor",
    page_icon="🎓",
    layout="wide"
)

# --- Helper Functions ---

@st.cache_resource
def load_model():
    """Loads the trained pipeline model."""
    if not os.path.exists("student_lifestyle_pipeline.pkl"):
        st.error("Model file 'student_lifestyle_pipeline.pkl' not found.")
        return None
    return joblib.load("student_lifestyle_pipeline.pkl")

@st.cache_data
def load_feature_lookup():
    """Loads the feature lookup CSV."""
    if not os.path.exists("feature_lookup.csv"):
        st.error("Feature lookup file 'feature_lookup.csv' not found.")
        return None
    # Ensure column names are stripped of whitespace
    df = pd.read_csv("feature_lookup.csv")
    df.columns = df.columns.str.strip()
    return df

def extract_categories_from_pipeline(pipeline, feature_names):
    """
    Attempts to extract learned categories from the OneHotEncoder in the pipeline.
    This allows us to show SelectBoxes instead of TextInputs for categorical data.
    """
    cat_options = {}
    try:
        # Access the preprocessor step
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Find the categorical transformer. 
        # Usually stored as ('cat', pipeline, columns) in ColumnTransformer
        # We search for the transformer named 'cat'
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'cat':
                # Inside the 'cat' pipeline, find the 'encoder' step
                encoder = transformer.named_steps['encoder']
                
                # Get the categories learned by OneHotEncoder
                # encoder.categories_ is a list of arrays, one for each column in 'cols'
                if hasattr(encoder, 'categories_'):
                    for col_name, categories in zip(cols, encoder.categories_):
                        # Clean up column name (model might expect 'f4', lookup has 'F4')
                        clean_col = col_name.lower() 
                        cat_options[clean_col] = list(categories)
    except Exception as e:
        # Fallback: If extraction fails (structure differs), we just return empty
        # and the UI will default to text inputs.
        pass
        
    return cat_options

# --- Main App Layout ---

def main():
    st.title("🎓 Student Lifestyle Predictor")
    st.markdown("""
    This application predicts a student's lifestyle score based on various academic 
    and personal habits. Enter the details below to get a prediction.
    """)

    # Load Data
    pipeline = load_model()
    lookup_df = load_feature_lookup()

    if pipeline is None or lookup_df is None:
        st.stop()

    # Extract known categories from the model to make the UI friendlier
    # We need to map feature codes (f4, f5) to these categories
    known_categories = extract_categories_from_pipeline(pipeline, lookup_df['feature_code'])

    # Form Container
    with st.form("prediction_form"):
        st.subheader("Student Details")
        
        input_data = {}
        
        # Create a grid layout for inputs
        cols = st.columns(3)
        
        # Iterate through the lookup CSV to generate inputs dynamically
        for index, row in lookup_df.iterrows():
            col_idx = index % 3
            
            feature_code_raw = row['feature_code'] # e.g., "F1"
            feature_name = row['relevance']        # e.g., "age"
            feature_type = row['type']             # e.g., "numeric"
            
            # The model expects lowercase keys (f1, f2...), derived from checking pipeline internals
            model_key = feature_code_raw.lower()
            
            with cols[col_idx]:
                label = f"{feature_name.title()} ({feature_code_raw})"
                
                if feature_type.strip() == 'numeric':
                    # Numeric Input
                    input_data[model_key] = st.number_input(
                        label, 
                        value=0.0, 
                        step=0.1,
                        key=model_key
                    )
                else:
                    # Categorical Input
                    # Check if we successfully extracted options from the model
                    options = known_categories.get(model_key)
                    
                    if options:
                        input_data[model_key] = st.selectbox(
                            label, 
                            options=options,
                            key=model_key
                        )
                    else:
                        # Fallback to text input if options aren't found
                        input_data[model_key] = st.text_input(
                            label,
                            placeholder="e.g., Yes, No, Male...",
                            key=model_key
                        )

        # Submit Button
        submit_btn = st.form_submit_button("Predict Lifestyle Score", type="primary")

    # --- Prediction Logic ---
    if submit_btn:
        try:
            # Create DataFrame from inputs
            input_df = pd.DataFrame([input_data])
            
            # Ensure columns are ordered correctly if the pipeline is strict
            # (Though ColumnTransformer usually handles by name)
            
            with st.spinner("Calculating..."):
                prediction = pipeline.predict(input_df)
                
                # Clip prediction as per the notebook logic
                final_score = np.clip(prediction[0], 0, 100)

            # Display Result
            st.markdown("---")
            st.subheader("Prediction Result")
            
            metric_col1, metric_col2 = st.columns([1, 2])
            
            with metric_col1:
                st.metric(
                    label="Predicted Score", 
                    value=f"{final_score:.2f} / 100"
                )
            
            with metric_col2:
                if final_score > 75:
                    st.success("🌟 Excellent! This student has a high predicted lifestyle score.")
                elif final_score > 50:
                    st.info("👍 Good. The lifestyle score is balanced.")
                else:
                    st.warning("⚠️ Attention Needed. The predicted score is on the lower side.")
                    
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info("Tip: Ensure categorical inputs match the values expected by the model (e.g., 'Male' vs 'M').")

if __name__ == "__main__":
    main()

