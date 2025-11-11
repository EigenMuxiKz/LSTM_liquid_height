import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from tensorflow.keras.models import load_model
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Four Tank System - LSTM Predictor",
    page_icon="💧",
    layout="wide"
)

# Title and description
st.title("🏭 Four Tank System - Liquid Height Predictor")
st.markdown("""
This application predicts the liquid heights in a four-tank system based on pump speeds.
Enter the pump speeds for both pumps and see how the liquid heights evolve over time.
""")

# Load model and artifacts
@st.cache_resource
def load_model_and_artifacts():
    try:
        model = load_model('lstm_four_tank_model.h5')
        with open('model_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return model, artifacts
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, artifacts = load_model_and_artifacts()
x_scaler = artifacts['x_scaler']
y_scaler = artifacts['y_scaler']
window = artifacts['window']
input_cols = artifacts['input_cols']
output_cols = artifacts['output_cols']

# Sidebar for user inputs
st.sidebar.header("⚙️ Control Parameters")
st.sidebar.markdown("### Pump Speeds")

v1 = st.sidebar.slider(
    "Pump 1 Speed (v1)",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1,
    help="Speed of pump 1 (units: arbitrary)"
)

v2 = st.sidebar.slider(
    "Pump 2 Speed (v2)",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1,
    help="Speed of pump 2 (units: arbitrary)"
)

st.sidebar.markdown("### Initial Tank Heights")
h1_init = st.sidebar.number_input("Tank 1 Initial Height (h1)", value=5.0, step=0.1)
h2_init = st.sidebar.number_input("Tank 2 Initial Height (h2)", value=5.0, step=0.1)
h3_init = st.sidebar.number_input("Tank 3 Initial Height (h3)", value=5.0, step=0.1)
h4_init = st.sidebar.number_input("Tank 4 Initial Height (h4)", value=5.0, step=0.1)

prediction_steps = st.sidebar.slider(
    "Prediction Time Steps",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
    help="Number of future time steps to predict"
)

# Prediction button
predict_button = st.sidebar.button("🚀 Predict Heights", type="primary", use_container_width=True)

# Main content
if predict_button:
    with st.spinner("Generating predictions..."):
        # Initialize the sequence with initial heights
        current_heights = np.array([h1_init, h2_init, h3_init, h4_init])
        
        # Create initial sequence (repeat initial state for window length)
        initial_input = np.array([[v1, v2, h1_init, h2_init, h3_init, h4_init]] * window)
        
        # Scale the initial input
        sequence = x_scaler.transform(initial_input)
        sequence = sequence.reshape(1, window, len(input_cols))
        
        # Store predictions
        predictions = [current_heights.copy()]
        
        # Iterative prediction
        for step in range(prediction_steps):
            # Predict next state
            pred_scaled = model.predict(sequence, verbose=0)
            
            # Inverse transform to get actual heights
            pred_heights = y_scaler.inverse_transform(pred_scaled)[0]
            predictions.append(pred_heights.copy())
            
            # Create next input: constant pump speeds + predicted heights
            next_input = np.array([v1, v2, pred_heights[0], pred_heights[1], 
                                  pred_heights[2], pred_heights[3]])
            
            # Scale the next input
            next_input_scaled = x_scaler.transform(next_input.reshape(1, -1))
            
            # Update sequence: remove oldest, add newest
            sequence = np.concatenate([
                sequence[:, 1:, :],
                next_input_scaled.reshape(1, 1, -1)
            ], axis=1)
        
        # Convert predictions to DataFrame
        predictions_array = np.array(predictions)
        df_predictions = pd.DataFrame(
            predictions_array,
            columns=['Tank 1 (h1)', 'Tank 2 (h2)', 'Tank 3 (h3)', 'Tank 4 (h4)']
        )
        df_predictions['Time Step'] = range(len(predictions))
        
        # Display results
        st.success("✅ Prediction Complete!")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Time Series Plot", "📈 Individual Tanks", "📋 Data Table"])
        
        with tab1:
            st.subheader("Liquid Heights Over Time")
            
            # Create interactive plot with all tanks
            fig = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for i, tank in enumerate(['Tank 1 (h1)', 'Tank 2 (h2)', 'Tank 3 (h3)', 'Tank 4 (h4)']):
                fig.add_trace(go.Scatter(
                    x=df_predictions['Time Step'],
                    y=df_predictions[tank],
                    mode='lines',
                    name=tank,
                    line=dict(width=2, color=colors[i])
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
        
        with tab2:
            st.subheader("Individual Tank Responses")
            
            # Create 2x2 subplot for individual tanks
            fig_sub = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Tank 1', 'Tank 2', 'Tank 3', 'Tank 4')
            )
            
            tanks = ['Tank 1 (h1)', 'Tank 2 (h2)', 'Tank 3 (h3)', 'Tank 4 (h4)']
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            
            for tank, (row, col), color in zip(tanks, positions, colors):
                fig_sub.add_trace(
                    go.Scatter(
                        x=df_predictions['Time Step'],
                        y=df_predictions[tank],
                        mode='lines',
                        name=tank,
                        line=dict(width=2, color=color),
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig_sub.update_xaxes(title_text="Time Step")
            fig_sub.update_yaxes(title_text="Height")
            fig_sub.update_layout(height=600, template='plotly_white')
            
            st.plotly_chart(fig_sub, use_container_width=True)
        
        with tab3:
            st.subheader("Prediction Data")
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Tank 1 Final Height",
                    f"{predictions_array[-1, 0]:.2f}",
                    f"{predictions_array[-1, 0] - predictions_array[0, 0]:.2f}"
                )
            
            with col2:
                st.metric(
                    "Tank 2 Final Height",
                    f"{predictions_array[-1, 1]:.2f}",
                    f"{predictions_array[-1, 1] - predictions_array[0, 1]:.2f}"
                )
            
            with col3:
                st.metric(
                    "Tank 3 Final Height",
                    f"{predictions_array[-1, 2]:.2f}",
                    f"{predictions_array[-1, 2] - predictions_array[0, 2]:.2f}"
                )
            
            with col4:
                st.metric(
                    "Tank 4 Final Height",
                    f"{predictions_array[-1, 3]:.2f}",
                    f"{predictions_array[-1, 3] - predictions_array[0, 3]:.2f}"
                )
            
            st.markdown("---")
            
            # Display data table
            st.dataframe(
                df_predictions.style.format({
                    'Tank 1 (h1)': '{:.3f}',
                    'Tank 2 (h2)': '{:.3f}',
                    'Tank 3 (h3)': '{:.3f}',
                    'Tank 4 (h4)': '{:.3f}'
                }),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = df_predictions.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"tank_predictions_v1_{v1}_v2_{v2}.csv",
                mime="text/csv"
            )

else:
    # Initial state - show instructions
    st.info("""
    👈 **Get Started:**
    1. Set the pump speeds (v1 and v2) in the sidebar
    2. Set the initial tank heights
    3. Choose the number of prediction steps
    4. Click the **Predict Heights** button to see the results
    """)
    
    # Show example visualization
    st.markdown("### 💡 About the Four Tank System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Inputs:**
        - **v1**: Pump 1 speed
        - **v2**: Pump 2 speed
        
        **Outputs:**
        - **h1**: Liquid height in Tank 1
        - **h2**: Liquid height in Tank 2
        - **h3**: Liquid height in Tank 3
        - **h4**: Liquid height in Tank 4
        """)
    
    with col2:
        st.markdown("""
        **How it works:**
        - The LSTM model predicts future liquid heights based on pump speeds
        - Predictions are made iteratively for the specified time horizon
        - The model uses the last 20 time steps to predict the next state
        - Results show how the system responds to the specified pump inputs
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Four Tank System LSTM Predictor | "
    "Built with Streamlit 🎈</div>",
    unsafe_allow_html=True
)
