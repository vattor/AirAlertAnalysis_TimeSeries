Ukraine Air Raid Alert: Time Series Classification

📌 Project Overview

This project is an MVP developed during a 2-day AI Summer School sprint. 
It applies advanced Time Series Analysis and Machine Learning to historical Ukrainian air raid alert data.
Rather than attempting the highly noisy task of predicting exact alert durations (Regression),
this project treats threat detection as a Binary Classification problem. 
It utilizes a Calibrated Random Forest to forecast the probability of an air raid alert occurring in a specific region within 15-minute intervals, 
acting as a simulated early-warning mechanism.
Dataset: Ukrainian Air Raid Sirens Dataset by Vadimkin: https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset

🧠 Methodology & Advanced Feature Engineering

To achieve high Receiver Operating Characteristic (ROC) AUC scores while maintaining practical Precision in highly imbalanced regions, this model implements several advanced data science techniques:
1. High-Fidelity Temporal Resolution (15-Min Blocks)
Raw alert timestamps were resampled into continuous 30-minute interval for the main region /15-minute intervals for the neighbour regions intervals.
This preserves the granular "momentum" of rapid-onset threats, preventing the catastrophic information loss that occurs when aggregating attacks into hourly blocks.

2. Deep Autoregressive Memory (96 Lags)

The model looks back at the target region's own status over the past 24 hours. This allows the Random Forest to capture diurnal patterns and predict the "All Clear" cascading signal when ongoing waves begin to subside.

3. Geospatial "Outer Perimeter" Awareness

Aerial threats take physical time to travel across the country. The pipeline automatically maps adjacent regions (neighbors) and extracts their historical data to act as an early warning system.
The "Simultaneous Echo" Trap: The model intentionally drops lag_1 (the immediate past 15 mins) for neighboring regions to prevent the algorithm from memorizing simultaneous nationwide alarms.
The Travel Window: The model strictly looks at neighbor lag_2, lag_3, and lag_4 (15 to 60 minutes ago), perfectly isolating the physical travel time of incoming threats across borders.

4. Cyclic Time Encoding

Linear time representations fail algorithms at midnight (the model mistakenly calculates that 23:00 and 00:00 are 23 hours apart instead of 1 hour). 
This project encodes hour_of_day and day_of_week into continuous circles using Sine and Cosine transformations, eliminating the midnight mathematical fracture.

5. Probability Calibration & Generalization

Imbalanced Data: Regions like Zhytomyr are calm 95% of the time. To prevent the model from lazily predicting 0, class_weight='balanced' was applied.
Isotonic Calibration: Because balanced weights distort internal probability outputs, the base Random Forest is wrapped in a CalibratedClassifierCV. 
This ensures the model outputs true, reliable risk percentages, allowing for a strict, actionable 30% User Warning Threshold.
Noise Reduction: min_samples_leaf=5 was implemented to prevent the deep decision trees from memorizing one-off radar glitches.

📊 Visual Outputs

When the script is executed, it generates three analytical dashboards:
ROC Curve: Demonstrates the model's overall mathematical ability to separate Calm vs. 
Alert states (typically achieving ~0.88 - 0.92 AUC).
Top 15 Feature Importances: Visualizes the decision-making weights, proving the integration of the Target's Autoregressive Lags and the Geospatial Neighbor Warnings.
Simulated Real-Time Forecast: A 48-hour timeline overlaying the model's continuous probability calculation against historical, actual air raid events.

⚙️ Installation & Usage
1. Clone the repository:
git clone [ https://github.com/vattor/AirAlertAnalysis_TimeSeries ] AirAlertAnalysis_TimeSeries
2. Install dependencies:
pip install pandas numpy matplotlib scikit-learn requests
3. Run the analysis:
The script is configured out-of-the-box to run a geospatial analysis on Dnipropetrovska oblast (You can change the region in the code).
python data/main.py

Note: The script features dynamic ETag caching and will automatically download or update the .csv database from GitHub only if changes are detected.

⚠️ Disclaimer

This project was developed strictly for educational purposes as part of an AI Summer School curriculum.
It relies on volunteer-gathered historical data and is NOT a life-safety application.
Do not use this algorithm for real-world threat detection or personal safety decisions.
