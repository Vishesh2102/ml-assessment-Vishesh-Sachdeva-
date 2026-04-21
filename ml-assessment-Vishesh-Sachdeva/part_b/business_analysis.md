# Part B: Business Case Analysis — Promotion Effectiveness

## B1. Problem Formulation

### (a) Formulation
* **Target Variable:** `items_sold` (Continuous numerical value).
* **Input Features:** Store location types, store size, local competition density, calendar flags (weekends/festivals), and the 5 specific promotion types.
* **Type of ML Problem:** **Supervised Regression**.
* **Justification:** The goal is to predict a specific quantity of goods sold to optimize inventory. Regression models are designed to output continuous numerical values, allowing for a precise calculation of expected demand for each promotion type rather than just a "high/low" classification.

### (b) Target Variable Reliability
Using `items_sold` (sales volume) is more reliable than total revenue because revenue is often distorted by external factors like price changes, inflation, or the sale of a few high-value luxury items that do not reflect actual customer footfall. Volume is a "purer" indicator of how many customers were actually moved to action by a promotion.
* **Broader Principle:** This illustrates **Proximal Target Selection**. In machine learning, it is critical to select a target variable that is most closely and directly influenced by the "lever" being tested (the promotion) rather than downstream financial metrics influenced by outside noise.

### (c) Alternative Modelling Strategy
I propose an **Ensemble of Local Models**. Instead of one global model, we should develop sub-models specifically for each location type (Urban, Semi-Urban, and Rural). Because the demographic behavior and logistical constraints differ vastly across these settings, local models can capture location-specific nuances—such as rural price sensitivity vs. urban convenience preference—that a global model would likely average out and ignore.

---

## B2. Data and EDA Strategy

### (a) Table Joins and Grain
* **Join Logic:** I would use a **Left Join** starting with the **Calendar Table** to create a continuous time-series "spine." I would then merge the Transaction, Store, and Promotion tables using `store_id` and `date` as keys.
* **Grain:** The final modelling dataset grain is **One row per Store-Day**.
* **Aggregations:** Transaction-level data should be aggregated to a daily sum of `items_sold` per store. Categorical attributes like location and promotion type should be encoded (e.g., One-Hot Encoding) to be readable by the regression algorithm.

### (b) EDA Strategy
1. **Correlation Heatmap:** To determine which store attributes (like size or competition) have the strongest linear relationship with sales volume.
2. **Promotion Efficiency Boxplots:** To visualize the spread and median of `items_sold` for each promotion type, identifying which offers have the most consistent "lift."
3. **Seasonality Analysis:** Plotting sales over time to identify baseline "festival" spikes. This helps ensure the model doesn't credit a promotion for a holiday rush that would have happened regardless.
4. **Outlier Detection:** Identifying stores with extreme sales spikes to determine if they represent a unique market behavior or simply data entry errors.

### (c) Handling Imbalance
The 80% "No Promotion" data creates a "majority class" bias where the model may struggle to accurately predict the impact of the 20% of active promotions. To address this, I would use **Cost-Sensitive Learning**, assigning a higher weight to promotional days during the model's loss calculation to ensure it prioritizes accuracy on the days that matter most to the marketing team.

---

## B3. Model Evaluation and Deployment

### (a) Train-Test Split and Metrics
* **Strategy:** A **Temporal (Chronological) Split** (e.g., training on the first 30 months, testing on the final 6 months).
* **Why not Random?** Shuffling retail data is inappropriate because it introduces "Look-Ahead Bias," where the model learns from future trends to predict past events—an impossible scenario in a real business environment.
* **Metrics:** **Mean Absolute Error (MAE)**. 
* **Interpretation:** MAE provides the average number of items our forecast is off by, which is easily interpreted by store managers for stock buffering. It translates directly into physical inventory units.

### (b) Feature Importance & Store 12
To explain the differing recommendations for Store 12, I would use **Feature Importance** analysis:
* **December:** The model likely identifies "Festival=True" as the highest-weight feature. It recommends a **Loyalty Bonus** because it recognizes that holiday footfall is already at its peak, and a heavy price discount is unnecessary to move inventory.
* **March:** In a month with no festivals, the model identifies "Demand" as the limiting factor. It recommends a **Flat Discount** as the most important feature to artificially lower the barrier to entry and clear seasonal stock.
* **Communication:** I would present a "Feature Impact" chart to show exactly how much "sales credit" was given to the calendar (natural demand) vs. the promotion itself (artificial demand).

### (c) Deployment and Monitoring
* **Process:** The final model and preprocessing pipeline would be serialized using `joblib` for deployment.
* **Monthly Cycle:** On the 1st of each month, the marketing team feeds the upcoming month's calendar and store parameters into the model. The model runs 5 simulations (one for each promotion) and outputs the predicted `items_sold` for each.
* **Monitoring:** I would implement **Performance Monitoring** to track the "Residual Error" (Predicted vs. Actual) in real-time. If the error exceeds a predefined threshold for two consecutive months, a "Model Decay" alert is triggered, signaling the need for retraining to account for new market trends.