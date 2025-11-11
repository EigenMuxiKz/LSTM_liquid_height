import os
# CRITICAL: Enable Keras 2 compatibility mode for TensorFlow 2.20+
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Four Tank System - LSTM Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .model-layer {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border-left: 4px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏭 Four Tank System - LSTM Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; color: #666;'>
Neural Network Demonstration for Chemical Process Control using LSTM
</p>
""", unsafe_allow_html=True)

# Load model and artifacts with detailed error reporting
@st.cache_resource
def load_model_and_artifacts():
    error_details = []
    
    try:
        st.info("Loading model files...")
        
        # Check if files exist
        import os
        files_needed = ['lstm_four_tank_model.h5', 'model_artifacts.pkl']
        missing_files = [f for f in files_needed if not os.path.exists(f)]
        
        if missing_files:
            error_msg = f"❌ Missing files: {', '.join(missing_files)}"
            st.error(error_msg)
            st.warning("""
            **Required files:**
            - `lstm_four_tank_model.h5` - The trained LSTM model
            - `model_artifacts.pkl` - Scalers and parameters
            
            Please make sure these files are in the same directory as the app.
            """)
            return None, None, error_msg
        
        # Try to load artifacts first (usually more reliable)
        try:
            with open('model_artifacts.pkl', 'rb') as f:
                artifacts = pickle.load(f)
            st.success("✅ Loaded model artifacts successfully")
        except Exception as e:
            error_msg = f"Failed to load artifacts: {str(e)}"
            st.error(error_msg)
            return None, None, error_msg
        
        # Try multiple methods to load the model
        model = None
        
        # Method 1: Load with compile=False
        try:
            st.info("Attempting Method 1: Load with compile=False...")
            model = tf.keras.models.load_model('lstm_four_tank_model.h5', compile=False)
            model.compile(optimizer='adam', loss='mse')
            st.success("✅ Model loaded using Method 1")
        except Exception as e1:
            error_details.append(f"Method 1 failed: {str(e1)}")
            
            # Method 2: Load with custom objects
            try:
                st.info("Attempting Method 2: Load with custom objects...")
                custom_objects = {'time_major': False}
                model = tf.keras.models.load_model('lstm_four_tank_model.h5', 
                                                   custom_objects=custom_objects,
                                                   compile=False)
                model.compile(optimizer='adam', loss='mse')
                st.success("✅ Model loaded using Method 2")
            except Exception as e2:
                error_details.append(f"Method 2 failed: {str(e2)}")
                
                # Method 3: Load weights only
                try:
                    st.info("Attempting Method 3: Rebuild architecture and load weights...")
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import LSTM, Dense, Dropout
                    
                    # Rebuild the architecture
                    model = Sequential([
                        LSTM(64, return_sequences=True, input_shape=(20, 6)),
                        Dropout(0.2),
                        LSTM(64, return_sequences=False),
                        Dropout(0.2),
                        Dense(4)
                    ])
                    
                    model.compile(optimizer='adam', loss='mse')
                    
                    # Try to load weights
                    model.load_weights('lstm_four_tank_model.h5')
                    st.success("✅ Model loaded using Method 3 (weights only)")
                except Exception as e3:
                    error_details.append(f"Method 3 failed: {str(e3)}")
                    
                    error_msg = "Failed to load model after trying all methods:\n" + "\n".join(error_details)
                    st.error(error_msg)
                    
                    # Show detailed error info
                    with st.expander("🔍 Detailed Error Information"):
                        st.code("\n\n".join(error_details))
                        st.markdown("""
                        **Possible solutions:**
                        1. Retrain the model with the current TensorFlow version
                        2. Make sure `tf_keras` is installed: `pip install tf_keras`
                        3. Check if the model file is corrupted
                        4. Verify TensorFlow version compatibility
                        """)
                    
                    return None, artifacts, error_msg
        
        return model, artifacts, None
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        st.error(error_msg)
        return None, None, error_msg

# Load model
model, artifacts, error = load_model_and_artifacts()

# If loading failed, show diagnostic info and stop
if model is None:
    st.error("❌ Failed to load the LSTM model")
    
    st.markdown("---")
    st.subheader("🔧 Troubleshooting Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Check your requirements.txt includes:**
        ```
        streamlit
        numpy<2.0.0
        pandas
        tensorflow>=2.15.0
        tf_keras
        scikit-learn
        plotly
        ```
        """)
        
        st.markdown("""
        **2. Verify files are uploaded:**
        - `lstm_four_tank_model.h5`
        - `model_artifacts.pkl`
        - `streamlit_app.py`
        """)
    
    with col2:
        st.markdown("""
        **3. Try retraining the model:**
        
        Use the provided `train_and_save_model.py` script with:
        ```python
        import os
        os.environ['TF_USE_LEGACY_KERAS'] = '1'
        ```
        at the very top.
        """)
        
        st.markdown("""
        **4. Check TensorFlow installation:**
        ```python
        import tensorflow as tf
        print(tf.__version__)
        ```
        Should show version 2.15.0 or higher.
        """)
    
    st.stop()

# Extract artifacts
x_scaler = artifacts['x_scaler']
y_scaler = artifacts['y_scaler']
window = artifacts['window']
input_cols = artifacts['input_cols']
output_cols = artifacts['output_cols']

st.success("✅ All files loaded successfully!")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🤖 Make Predictions", "🧠 Model Architecture", "📊 Model Info", "📚 About"])

with tab1:
    st.header("🎯 LSTM Predictions")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        
        st.markdown("### Pump Speeds")
        v1 = st.slider("Pump 1 Speed (v1)", 0.0, 10.0, 5.0, 0.1)
        v2 = st.slider("Pump 2 Speed (v2)", 0.0, 10.0, 5.0, 0.1)
        
        st.markdown("### Initial Tank Heights")
        h1_init = st.number_input("Tank 1 Initial Height (h1)", value=5.0, step=0.1)
        h2_init = st.number_input("Tank 2 Initial Height (h2)", value=5.0, step=0.1)
        h3_init = st.number_input("Tank 3 Initial Height (h3)", value=5.0, step=0.1)
        h4_init = st.number_input("Tank 4 Initial Height (h4)", value=5.0, step=0.1)
        
        prediction_steps = st.slider("Prediction Time Steps", 10, 200, 50, 10)
        predict_button = st.button("🚀 Predict Heights", type="primary", use_container_width=True)
    
    with col2:
        if predict_button:
            with st.spinner("Generating predictions..."):
                try:
                    current_heights = np.array([h1_init, h2_init, h3_init, h4_init])
                    initial_input = np.array([[v1, v2, h1_init, h2_init, h3_init, h4_init]] * window)
                    
                    sequence = x_scaler.transform(initial_input)
                    sequence = sequence.reshape(1, window, len(input_cols))
                    
                    predictions = [current_heights.copy()]
                    
                    for step in range(prediction_steps):
                        pred_scaled = model.predict(sequence, verbose=0)
                        pred_heights = y_scaler.inverse_transform(pred_scaled)[0]
                        predictions.append(pred_heights.copy())
                        
                        next_input = np.array([v1, v2, pred_heights[0], pred_heights[1], 
                                              pred_heights[2], pred_heights[3]])
                        next_input_scaled = x_scaler.transform(next_input.reshape(1, -1))
                        
                        sequence = np.concatenate([
                            sequence[:, 1:, :],
                            next_input_scaled.reshape(1, 1, -1)
                        ], axis=1)
                    
                    predictions_array = np.array(predictions)
                    df_predictions = pd.DataFrame(
                        predictions_array,
                        columns=['Tank 1 (h1)', 'Tank 2 (h2)', 'Tank 3 (h3)', 'Tank 4 (h4)']
                    )
                    df_predictions['Time Step'] = range(len(predictions))
                    
                    st.success("✅ Prediction Complete!")
                    
                    # Display final values
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                    
                    for i, (col, tank, color) in enumerate(zip([col_m1, col_m2, col_m3, col_m4], 
                                                                ['Tank 1', 'Tank 2', 'Tank 3', 'Tank 4'], 
                                                                colors)):
                        with col:
                            st.markdown(f"""
                            <div class="metric-card" style="border-left-color: {color};">
                                <h4 style="color: {color}; margin: 0;">{tank}</h4>
                                <h2 style="margin: 0.5rem 0;">{predictions_array[-1, i]:.2f}</h2>
                                <small>Change: {predictions_array[-1, i] - predictions_array[0, i]:+.2f}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Plot
                    st.subheader("Liquid Heights Over Time")
                    fig = go.Figure()
                    
                    for i, (tank, color) in enumerate(zip(['Tank 1 (h1)', 'Tank 2 (h2)', 'Tank 3 (h3)', 'Tank 4 (h4)'], colors)):
                        fig.add_trace(go.Scatter(
                            x=df_predictions['Time Step'],
                            y=df_predictions[tank],
                            mode='lines',
                            name=tank,
                            line=dict(width=3, color=color)
                        ))
                    
                    fig.update_layout(
                        title=f"Four Tank System Response (v1={v1}, v2={v2})",
                        xaxis_title="Time Step",
                        yaxis_title="Liquid Height",
                        hovermode='x unified',
                        height=500,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    csv = df_predictions.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name=f"tank_predictions_v1_{v1}_v2_{v2}.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.code(str(e))
        else:
            st.info("👈 Set the parameters and click **Predict Heights** to see results")

with tab2:
    st.header("🧠 LSTM Model Architecture")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Model Summary")
        
        try:
            total_params = model.count_params()
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Parameters</h3>
                <h2>{total_params:,}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 🔗 Layer Architecture")
            
            for i, layer in enumerate(model.layers):
                layer_config = layer.get_config()
                layer_name = layer.__class__.__name__
                
                if 'LSTM' in layer_name:
                    units = layer_config.get('units', 'N/A')
                    return_seq = layer_config.get('return_sequences', False)
                    
                    st.markdown(f"""
                    <div class="model-layer">
                        <strong>Layer {i+1}: LSTM</strong><br>
                        🔸 Units: {units}<br>
                        🔸 Return Sequences: {return_seq}<br>
                        🔸 Output Shape: {layer.output_shape}
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif 'Dropout' in layer_name:
                    rate = layer_config.get('rate', 'N/A')
                    
                    st.markdown(f"""
                    <div class="model-layer">
                        <strong>Layer {i+1}: Dropout</strong><br>
                        🔸 Rate: {rate}
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif 'Dense' in layer_name:
                    units = layer_config.get('units', 'N/A')
                    activation = layer_config.get('activation', 'N/A')
                    
                    st.markdown(f"""
                    <div class="model-layer">
                        <strong>Layer {i+1}: Dense (Output)</strong><br>
                        🔸 Units: {units}<br>
                        🔸 Activation: {activation}<br>
                        🔸 Output Shape: {layer.output_shape}
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Could not display architecture: {str(e)}")
    
    with col2:
        st.subheader("📐 Architecture Visualization")
        
        st.markdown("""
        ```
        ┌─────────────────────────────┐
        │   Input Layer (20, 6)       │
        │   [v1, v2, h1, h2, h3, h4]  │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   LSTM Layer 1 (64 units)   │
        │   Return Sequences: True    │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   Dropout (0.2)             │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   LSTM Layer 2 (64 units)   │
        │   Return Sequences: False   │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   Dropout (0.2)             │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   Dense Layer (4 units)     │
        │   Output: [h1, h2, h3, h4]  │
        └─────────────────────────────┘
        ```
        """)
        
        st.markdown("---")
        st.subheader("🔍 Model Details")
        
        st.markdown(f"""
        **Input Configuration:**
        - Window Size: {window} time steps
        - Features per step: {len(input_cols)}
        - Input Shape: (batch, {window}, {len(input_cols)})
        
        **Output Configuration:**
        - Predicted Variables: {len(output_cols)}
        - Output Names: {', '.join(output_cols)}
        - Output Shape: (batch, {len(output_cols)})
        """)

with tab3:
    st.header("📊 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Capabilities")
        st.markdown("""
        - Multi-step ahead forecasting
        - Multivariate input/output
        - Captures system dynamics
        - Handles nonlinear behavior
        """)
    
    with col2:
        st.subheader("🔬 LSTM Benefits")
        st.markdown("""
        - Remembers past states
        - Learns temporal patterns
        - Processes sequences
        - Prevents vanishing gradients
        """)

with tab4:
    st.header("📚 About")
    st.markdown("""
    This application demonstrates LSTM neural networks for predicting liquid heights 
    in a four-tank system - a classic chemical engineering control problem.
    
    **Features:**
    - Real-time predictions
    - Interactive visualizations
    - Model architecture display
    - Educational content
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🏭 Four Tank System LSTM Predictor | Built with Streamlit 🎈
</div>
""", unsafe_allow_html=True)

