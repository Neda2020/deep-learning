# Deep Learning Challenge: Predicting Funding Success for Alphabet Soup

## Overview
The nonprofit foundation **Alphabet Soup** seeks a machine learning tool to **predict funding success** for applicants. This project uses **deep learning and neural networks** to build a binary classification model that determines whether an applicant is likely to use funding effectively.

The dataset contains **34,000+ records** of past applicants with metadata such as **application type, affiliation, classification, funding request amount, and success status**.

---

## 📂 Files in This Repository
- `AlphabetSoupCharity.ipynb` → **Main deep learning model**
- `AlphabetSoupCharity_Optimization.ipynb` → **Optimized version of the model**
- `AlphabetSoupCharity.h5` → **Trained model (HDF5 file)**
- `AlphabetSoupCharity_Optimization.h5` → **Optimized trained model**
- `charity_data.csv` → **Dataset**
- `README.md` → **Project overview & instructions**

---

## 🚀 Step 1: Data Preprocessing

The dataset is preprocessed using **Pandas** and **scikit-learn**:
1. **Loaded the dataset** from a cloud URL (`charity_data.csv`).
2. **Identified target (`IS_SUCCESSFUL`) and feature variables**.
3. **Dropped non-essential columns** (`EIN`, `NAME`).
4. **Checked unique values in categorical columns**.
5. **Combined rare categories** under a new value `"Other"`.
6. **Used `pd.get_dummies()`** for categorical encoding.
7. **Split the data** into **training (80%)** and **testing (20%)** sets.
8. **Applied `StandardScaler()`** to normalize numerical features.

---

## 🧠 Step 2: Compile, Train, and Evaluate the Model

A **deep neural network (DNN)** was built using **TensorFlow/Keras**:
1. **Defined a Sequential model** with:
   - **First hidden layer** → 80 neurons, ReLU activation.
   - **Second hidden layer** → 30 neurons, ReLU activation.
   - **Output layer** → 1 neuron, Sigmoid activation (for binary classification).
2. **Compiled the model** using:
   - `optimizer="adam"`
   - `loss="binary_crossentropy"`
   - `metrics=["accuracy"]`
3. **Trained the model** with:
   - `epochs=50`
   - `batch_size=32`
4. **Saved the trained model** as `AlphabetSoupCharity.h5`.

**Evaluation Results:**
```python
model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
✅ Achieved an accuracy of ~72% (before optimization).
________________________________________
🛠 Step 3: Model Optimization
To improve accuracy above 75%, several optimizations were applied:
1.	Adjusted input data:
o	Removed additional irrelevant features.
o	Created more bins for rare categorical values.
2.	Modified neural network architecture:
o	Increased neurons in hidden layers.
o	Added an extra hidden layer.
o	Experimented with different activation functions.
3.	Tuned training parameters:
o	Increased epochs to 100.
o	Adjusted batch_size for better convergence.
🔹 The optimized model was trained and saved as AlphabetSoupCharity_Optimization.h5.
________________________________________
📊 Step 4: Model Performance Analysis
Data Preprocessing
✅ Target Variable: IS_SUCCESSFUL
✅ Feature Variables: APPLICATION_TYPE, CLASSIFICATION, USE_CASE, ASK_AMT, etc.
✅ Dropped Columns: EIN, NAME
Neural Network Architecture
Layer	Neurons	Activation
Input Layer	43	-
Hidden Layer 1	80	ReLU
Hidden Layer 2	30	ReLU
Output Layer	1	Sigmoid
Model Performance
•	Baseline Model Accuracy: ~72%
•	Optimized Model Accuracy: 75%+ 🚀
Optimization Techniques Used
✅ Increased neurons in hidden layers
✅ Added a third hidden layer
✅ Adjusted training epochs and batch size
✅ Used different activation functions
________________________________________
📌 Step 5: Model Deployment & Future Improvements
The trained model is exported and can be reloaded for further analysis:
python
CopyEdit
from tensorflow.keras.models import load_model
loaded_model = load_model("AlphabetSoupCharity_Optimization.h5")
Future Recommendations
•	Try Random Forest or XGBoost for comparison.
•	Perform hyperparameter tuning using GridSearchCV.
•	Use AutoML (like TensorFlow AutoKeras) for automated optimization.
________________________________________
💾 Instructions to Run the Project
1️⃣ Clone the Repository
sh
CopyEdit
git clone https://github.com/your-username/deep-learning-challenge.git
cd deep-learning-challenge
2️⃣ Install Dependencies
sh
CopyEdit
pip install pandas scikit-learn tensorflow
3️⃣ Run the Jupyter Notebook
sh
CopyEdit
jupyter notebook
Open AlphabetSoupCharity.ipynb and execute the cells.
4️⃣ Load and Evaluate the Saved Model
python
CopyEdit
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("AlphabetSoupCharity_Optimization.h5")

# Evaluate model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Model Accuracy: {accuracy:.2%}")
________________________________________
📜 Summary
This project successfully developed a deep learning model to predict funding success for Alphabet Soup applicants.
🔹 Baseline Accuracy: ~72%
🔹 Optimized Model Accuracy: 75%+
🔹 Potential Future Enhancements: Hyperparameter tuning, different models (Random Forest, XGBoost).
🚀 This project demonstrates the power of deep learning for real-world classification problems!
________________________________________

•	Dataset: Provided by Alphabet Soup.
•	Libraries Used: Pandas, Scikit-Learn, TensorFlow/Keras.
•	Developed by: Neda Jamal
🔗 GitHub Repository: deep-learning-challenge





