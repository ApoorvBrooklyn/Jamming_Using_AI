import streamlit as st
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import json
import os
from data_generator import SignalGenerator, SignalParams
import datetime

def load_trained_model(model_dir):
    """Load the trained model with proper configuration"""
    try:
        # Load model configuration
        config_path = os.path.join(model_dir, 'model_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load the model
        model_path = os.path.join(model_dir, 'model.h5')
        model = load_model(model_path)
        
        return model, config['input_shape']
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model is trained before running inference.")
        return None, None

def analyze_signal(signal, model, input_shape):
    """Analyze signal using the trained model"""
    try:
        # Reshape signal for model input
        signal_reshaped = signal.reshape(1, input_shape[0], 1)
        
        # Get model prediction
        prediction = model.predict(signal_reshaped, verbose=0)[0][0]
        return float(prediction)
    except Exception as e:
        st.error(f"Error during signal analysis: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Signal Jamming Detection System", layout="wide")
    
    st.title("Signal Jamming Detection System")
    
    # Load trained model
    model, input_shape = load_trained_model("models")
    
    if model is None:
        st.warning("Please train the model first by running model_trainer.py")
        return
    
    # Sidebar parameters
    st.sidebar.header("Signal Parameters")
    signal_type = st.sidebar.selectbox("Signal Type", ["PSK", "QAM"])
    frequency = st.sidebar.slider("Carrier Frequency (Hz)", 10, 200, 50)
    amplitude = st.sidebar.slider("Signal Amplitude", 0.1, 2.0, 1.0)
    
    st.sidebar.header("Jamming Parameters")
    noise_type = st.sidebar.selectbox("Jamming Type", ["gaussian", "pulse", "swept"])
    noise_level = st.sidebar.slider("Jamming Intensity", 0.1, 3.0, 0.5)
    
    # Initialize signal generator
    generator = SignalGenerator()
    params = SignalParams(frequency=frequency, amplitude=amplitude)
    
    try:
        # Generate signals
        if signal_type == "PSK":
            clean_signal = generator.generate_psk(params)
        else:
            clean_signal = generator.generate_qam(params)
        
        # Add jamming
        jammed_signal = generator.add_jamming(clean_signal, noise_type, noise_level)
        
        # Analyze signal
        jamming_probability = analyze_signal(jammed_signal, model, input_shape)
        
        if jamming_probability is not None:
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Signal Visualization")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=clean_signal, name="Clean Signal",
                                       line=dict(color='blue')))
                fig.add_trace(go.Scatter(y=jammed_signal, name="Jammed Signal",
                                       line=dict(color='red')))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Jamming Analysis Results")
                st.metric("Jamming Probability", f"{jamming_probability:.2%}")
                st.metric("Signal Power", f"{np.mean(clean_signal**2):.2f} dB")
                st.metric("Noise Power", 
                         f"{np.mean((jammed_signal-clean_signal)**2):.2f} dB")
                snr = 10 * np.log10(np.mean(clean_signal**2) / 
                                   np.mean((jammed_signal-clean_signal)**2))
                st.metric("SNR", f"{snr:.2f} dB")
    
    except Exception as e:
        st.error(f"Error processing signal: {str(e)}")
    
    # Add timestamp
    st.sidebar.markdown("---")
    st.sidebar.text(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()