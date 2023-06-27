# Problem Statement
Predictive maintenance is based on the collection, management, and intelligent use of data that focuses on the machine's condition and determine whether repair or other services are needed. It is designed to predict when the maintenance should be performed based on the current situation of the machine or asset so that early repair or maintenance can be conducted to reduce cost. It also allow for safety compliance, preemptive corrective actions and increased asset life by looking ahead and knowing when the failure is likely to happen. 

# Synthetically-Generated Dataset (from Kaggle)
Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

# Dataset Description
Source: [Domain Knowledge Regarding the Dataset](https://www.kaggle.com/competitions/playground-series-s3e17/discussion/416765)
- `UDI`, integer to identify each device.
- `Product Id`, a unique identifier that combine the `Type` variable followed by a number identifier.
- `Type`, the type of the product or device (L/M/H).
- `Air Temperature [K]`, float number to represent the air temperature in Kelvin.
- `Process Temperature [K]`, float number to represent the process temperature in Kelvin.
- `Rotational speed [rpm]`, speed in rotations per minute that is calculated with the power of 2860W.
- `Torque`, float number to measure the torque in Nm (Newton Meter).
- `Tool Wear`, Time unit needed to wear down the product/tool.
- `Machine failure`, the binary target variable indicating the machine has failed or not. `0` represents No while `1` represents Yes.
- `TWF`, tool wear failure which is a binary variable, indicating industrial tool failure resulting in the need for equipment change and defective products.
- `HDF`, heat dissipation failure which is a binary variable, indicating failure in heat dissipation during the production process.
- `PWF`, power failure which is a binary variable, indicating that the power supplied was not fit to the production process need resulting in a failure.
- `OSF`, overstain failure which is a binary variable indicating failure involves product overstains which may be the result of high load and tension during production.
- `RNF`, random failure which is a binary variable, indicating that a random error causes the failure.

# Exploratory Data Analysis
- Target variable (`Machine failure`) and other binary variables has extremely imbalanced class distribution where almost 99% of the values are `0`.
- The variables `Air temperature [K]`, `Process temperature [K]`, and `Torque [Nm]` follows the pattern of normal distribution while the `Rotational speed [rpm]` has right-skewed distribution. The `Tool wear [min]` has pattern where the values of the variables are close to each other but dropped significantly after value of `200`.
- Based on the variables `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`, when the value of these variables is `1`, then the occurrences of machine failure drop significantly as compared to when the value is `0`.

# Data Preprocessing Steps
1. Conduct feature engineering to generate new features based on the existing columns.
2. Normalize the numerical variables with `MinMaxScaler()` (Min-max normalization) from `sklearn` library.
3. Integer encode the categorical variable using `LabelEncoder()` from `sklearn` library.
4. Replaces columns' names with appropriate and consistent naming. 
5. Partition data into training and testing set according to the ratio of `8:2`.
# Models
- Neural network
- Ridge Classifier
- Random Forest
- XGBoost Classifier
- Logistic Regression
- Light Gradient Boosting Machine (LGBM)
- Hybrid

# Model Evaluation
Neural network performs the best with precision of 0.99, recall of 0.76, and F1-Score of 0.86. Overall, the Area Under the Curve (AUC) score of the Neural Network model is **0.947** (on training set) and **0.95787** on the testing set (public score on Kaggle).

# Limitation
The limitation is that the false negative is still very high which can be a major problem. Data-resampling methods such as **SMOTE** and **undersampling** have been used to resolve the imbalanced class set. However, this method does not improve the performance of the model but also increases the bias of the model towards positive class. Therefore, this method is not implemented in the notebook.

# Suggestion
Future dataset can include more samples from the positive classes so that the model will not have bias towards the majority class for better recall score.
