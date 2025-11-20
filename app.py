import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from lightgbm import LGBMRegressor

# Set page configuration
st.set_page_config(page_title="Relationship Probability Predictor", layout="wide")

st.title("💘 Relationship Probability Predictor")
st.markdown("""
This app predicts the probability of a relationship based on student lifestyle data.
It uses a **pre-trained model** loaded from a pickle file.
""")

# --- 1. Data Loading (For UI Population Only) ---
@st.cache_data
def load_reference_data(file):
    """
    Loads the training data strictly to populate dropdown options
    and determine min/max values for the input form.
    """
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()
    
    # Define useless columns to drop (same as training config)
    useless_cols = [
        'school_board', 'coffee_cups_daily', 'gym_frequency', 
        'passion_drive', 'home_state'
    ]
    target_col = 'relationship_probability'
    
    # Identify ID column
    possible_ids = [c for c in df.columns if 'id' in c]
    id_col = possible_ids[0] if possible_ids else df.columns[0]
    
    # Columns to exclude
    cols_to_exclude = [target_col, id_col] + useless_cols
    
    # Type casting for specific categorical columns
    cols_to_str = ['food_orders', 'memes_shared', 'clubs_joined', 'num_sports']
    for col in cols_to_str:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Filter Feature DataFrame
    X = df.drop(cols_to_exclude, axis=1, errors='ignore')
    
    # Identify Feature Types
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return X, num_cols, cat_cols

# --- 2. Load Pre-trained Model ---
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model file: {e}")
        return None

# --- 3. Sidebar Configuration ---
st.sidebar.header("Configuration")

# Load Model
model_path = "student_lifestyle_pipeline.pkl"
uploaded_model = st.sidebar.file_uploader("Upload Model (.pkl)", type=["pkl"])
if uploaded_model:
    pipeline = load_model(uploaded_model)
    st.sidebar.success("Custom model loaded!")
else:
    try:
        pipeline = load_model(model_path)
        st.sidebar.success(f"Loaded local model: {model_path}")
    except FileNotFoundError:
        st.sidebar.error("Model file not found.")
        pipeline = None

# Load Reference Data
uploaded_data = st.sidebar.file_uploader("Upload Reference Data (train.csv)", type=["csv"])
default_data_path = "train.csv"

X_reference = None
num_features = []
cat_features = []

if uploaded_data:
    X_reference, num_features, cat_features = load_reference_data(uploaded_data)
else:
    try:
        X_reference, num_features, cat_features = load_reference_data(default_data_path)
    except FileNotFoundError:
        st.sidebar.warning("Reference data (train.csv) not found. Upload it to populate form options.")

# --- 4. Input Form & Prediction ---

if pipeline is not None and X_reference is not None:
    st.header("Enter Student Details")
    
    with st.form("prediction_form"):
        input_data = {}
        
        # Categorical Inputs
        st.subheader("Categorical Attributes")
        c_cols = st.columns(3)
        for i, col_name in enumerate(cat_features):
            with c_cols[i % 3]:
                options = sorted(X_reference[col_name].unique().tolist())
                input_data[col_name] = st.selectbox(f"{col_name.replace('_', ' ').title()}", options)
        
        st.write("---")
        
        # Numerical Inputs
        st.subheader("Numerical Attributes")
        n_cols = st.columns(3)
        for i, col_name in enumerate(num_features):
            with n_cols[i % 3]:
                min_val = float(X_reference[col_name].min())
                max_val = float(X_reference[col_name].max())
                mean_val = float(X_reference[col_name].mean())
                
                input_data[col_name] = st.number_input(
                    f"{col_name.replace('_', ' ').title()}", 
                    min_value=0.0,
                    max_value=max_val * 2.0, # Allow some buffer
                    value=mean_val
                )
                
        submit = st.form_submit_button("Predict Probability")

    # --- 5. Prediction Logic ---
    if submit:
        # Convert input dict to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure consistency in types for the specific categorical columns
        cols_to_str = ['food_orders', 'memes_shared', 'clubs_joined', 'num_sports']
        for col in cols_to_str:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str)

        try:
            with st.spinner("Calculating..."):
                prediction = pipeline.predict(input_df)[0]
                prediction = np.clip(prediction, 0, 100)
            
            st.markdown("---")
            st.subheader("Result")
            
            metric_col1, metric_col2 = st.columns([1, 3])
            
            with metric_col1:
                st.metric(label="Relationship Probability", value=f"{prediction:.2f}%")
            
            with metric_col2:
                st.progress(int(prediction))
                if prediction < 30:
                    st.info("Low Probability")
                elif prediction < 70:
                    st.warning("Moderate Probability")
                else:
                    st.success("High Probability")
                    
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.write("Debug Info - Input Data Types:")
            st.write(input_df.dtypes)

elif pipeline is None:
    st.warning("Please upload or place 'student_lifestyle_pipeline.pkl' in the directory.")
elif X_reference is None:
    st.warning("Please upload 'train.csv' so the app knows what options to show in the form.")