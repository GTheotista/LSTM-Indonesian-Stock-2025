# LSTM Stock Price Prediction

## **Project Overview**
Stock price prediction is a crucial aspect of financial market analysis. This project utilizes a Long Short-Term Memory (LSTM) model to predict stock prices based on historical data. The objective is to analyze trends and provide insights into potential investment decisions.

## **Dataset**
The dataset consists of daily closing prices for eight selected stocks from different industry sectors, collected from Yahoo Finance. The data spans from **January 1, 2015, to January 1, 2025**.

### **Selected Stocks:**
| Industry Sector           | Stock Ticker |
|---------------------------|-------------|
| Consumer Cyclical        | ACES.JK     |
| Energy                  | ADRO.JK     |
| Consumer Defensive      | AMRT.JK     |
| Basic Materials         | ANTM.JK     |
| Financial Services      | BBCA.JK     |
| Communication Services  | TLKM.JK     |
| Healthcare             | KLBF.JK     |
| Technology             | GOTO.JK     |

## **Tools Used**
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, TensorFlow, Keras, Scikit-learn
- **Visualization:** Matplotlib, Seaborn

## **Project Workflow**
### **1. Exploratory Data Analysis (EDA)**
- **Statistical Summary:** Examined mean, standard deviation, min-max values, and distribution of stock prices.
- **Trend Analysis:** Visualized stock price movements over time.
- **Correlation Analysis:** Generated a heatmap to identify relationships among selected stocks.
- **Findings:**
  - Stock price movements were generally stable, with no significant upward or downward trends visible from the raw visualizations.
  - Some stocks exhibited high positive or negative correlations, indicating potential sectoral dependencies.

### **2. Data Preprocessing**
- **Normalization:** Applied MinMaxScaler to scale stock prices between [0,1].
- **Sequence Data Creation:** Used a **50-day window size** to transform time series data into sequential format for LSTM input.
- **Train-Test Split:** The dataset was split into **80% training and 20% testing**.

### **3. LSTM Model Architecture**
- **Layer 1:** LSTM layer with **50 units** (return sequences = True)
- **Dropout:** 20% to reduce overfitting
- **Layer 2:** LSTM layer with **50 units**
- **Dropout:** 20%
- **Dense Layer:** 25 neurons
- **Output Layer:** 1 neuron (predicting next day's price)
- **Optimizer:** Adam (learning rate = 0.001)
- **Loss Function:** Mean Squared Error (MSE)
- **Training Parameters:** 50 epochs, batch size = 32

### **4. Model Evaluation**
The model was evaluated using multiple metrics:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **Mean Absolute Percentage Error (MAPE)**
- **Directional Accuracy (DA)**: Measures the percentage of correctly predicted stock price movements.

#### **Decision Criterion:**
- **Directional Accuracy (DA) was chosen as the primary metric** for investment decisions, as predicting the correct movement (up/down) is more relevant than minimizing absolute errors.

### **5. Results & Conclusion**
Based on the prediction results, the top three stocks with the lowest MAPE and highest Directional Accuracy (DA) are BBCA, KLBF, and ANTM.
- **KLBF** is the most reliable choice, with the lowest MAPE of 3.40% and strong trend consistency.
- **BBCA** follows closely, showing stable growth potential with a MAPE of 1.73%.
- **ACES** demonstrates a relatively constant trend with a MAPE of 2.50%, indicating limited growth potential.

This model can be further improved with hyperparameter tuning, feature engineering, and the inclusion of external factors such as market indices or economic indicators.

---
**Author:** [Giovanny Theotista]

