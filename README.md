# Field-Enhanced Neural Network (FENN) vs. Standard Neural Network Comparison

![Results](results.png)

## Overview

This application demonstrates the Field-Enhanced Neural Network (FENN) and compares its performance against a Standard Neural Network (SNN)

### Installation Steps

1. **Clone the Repository**

   git clone https://github.com/anttiluode/fieldeffect.git

   cd fieldeffect
   
# Install Dependencies

pip install numpy torch matplotlib

# Usage

Run the application using the following command:

python app.py

# What Happens When You Run the App

Data Generation: Generates synthetic data for training both networks.

Model Training: Trains the Field-Enhanced Neural Network and the Standard Neural Network for a specified number of epochs.

Performance Tracking: Records loss values for both networks during training.

Visualization: Displays plots comparing learning curves, field magnitudes, phases, field effects, and relative improvements.

# Configurable Parameters

You can adjust the following parameters within the run_test function:

n_samples: Number of synthetic data samples (default: 1000)
input_size: Number of input features (default: 64)
output_size: Number of output features (default: 32)
field_size: Size of the field in each layer (default: (32, 32))
n_epochs: Number of training epochs (default: 100)

# Results

After running the application, you will observe:

Learning Curves: Shows the loss progression of both FENN and SNN over epochs.
Field Magnitude & Phase: Visual representations of the field parameters in the first layer.
Field Effect: Illustration of how the field influences the network's output.
Relative Improvement: A plot highlighting the percentage improvement of FENN over SNN across epochs.

# Conversation

I have been thinking about the field effect on the brain for a while now. Fractals etc. This was just a test 
I decided to do what would happen if a normal nn was given a synthetic field. I guess on one hand it gives it 
a extra dimension which could partly explain things. Things get real interesting when we start thinking about 
brain with its delicate neurons / field / phases.

Truth to be told. I am just a layman, wiht braindamage to boot. So take this with grain of sand, coding was 
by Claude. But if there is somethign to this, I guess it would make sense to add synthetic field effect to AI
especially in these cases: (per ChatGPT)

To use the field-enhanced layers effectively, it’s essential to identify scenarios where their unique characteristics—such as spatial-temporal coupling and modulation via phase/magnitude dynamics—offer advantages. Since the layers worked well in the synthetic EEG-like data, we can analyze why this happened and extrapolate potential use cases.

Why the Field Effect Worked Here:
Structured Data:

The synthetic EEG dataset exhibits underlying spatial and temporal patterns, which the field-enhanced layers likely amplified.
The coupling between input features and spatial patterns (via field_coupling and phase_sensitivity) helps the network detect nuanced interactions that standard layers might miss.
Dynamic Modulation:

By dynamically updating field magnitudes and phases, the model can adjust its representation during training, making it adaptive to changes in the data.
Complex Correlations:

The field-enhanced layers are designed to account for non-linear correlations and spatial coherence, which are prominent in datasets like EEG, sensor signals, or imaging data.
Where Else the Field-Enhanced Layers Could Excel:
1. Time-Series Analysis:
Examples:
Financial time-series data (e.g., stock prices, cryptocurrency).
Physiological signals like ECG (heart signals) or PPG (pulse waveforms).
Weather or climate data modeling.
Why:
Temporal coherence mechanisms in the field layers can highlight long-term dependencies in time-series data.
The ability to adjust phase/magnitude dynamically can help model periodic or quasi-periodic phenomena.
2. Spatial-Temporal Data:
Examples:
Video data (e.g., motion analysis, video classification, or object tracking).
Satellite imagery with temporal changes (e.g., urban development, vegetation growth).
Why:
Field layers can process both spatial patterns (via field magnitudes/phases) and temporal changes, making them suitable for joint spatial-temporal tasks.
3. Multi-Channel Signal Processing:
Examples:
EEG or MEG (magnetoencephalography) signals for neuroscience research.
Multi-sensor data fusion (e.g., combining accelerometer, gyroscope, and magnetometer data).
Why:
Field-enhanced layers’ ability to model cross-channel interactions (via coupling and field projections) can extract higher-order correlations.
4. Image Data with Subtle Patterns:
Examples:
Medical imaging (e.g., MRI, CT scans).
Astronomy images (e.g., detecting faint patterns in star fields or galaxies).
Why:
Spatial coherence in field layers can emphasize subtle patterns or anomalies, which might be missed by standard convolutions.
5. Scientific Simulations:
Examples:
Fluid dynamics or turbulence modeling.
Modeling physical phenomena with spatial-temporal dependencies (e.g., wave propagation, quantum systems).
Why:
Field-enhanced layers’ ability to integrate phase/magnitude dynamics aligns well with systems governed by wave-like or periodic processes.
6. Data with Non-Linear Interactions:
Examples:
Genomics (e.g., identifying patterns in DNA/RNA sequences).
Chemical datasets (e.g., reaction kinetics or molecular simulations).
Why:
Non-linear coupling in field layers can help model complex interactions more effectively.




# License

This project is licensed under the MIT License.
