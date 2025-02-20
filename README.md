# The Term Project for the Python Programming Course AIML-1213
## Data

Data itself is raw, unorganized facts. It can be simple and random.

When an observation is recorded, it needs to be processed. This process is for the data to be transformed to fit a model for analysis or handling missing data field. Even though errors are mych worse than missing ones.

## Data Preparation

For a desired outcome,

- Features need to be selected
- Clean the data (e.g. throw out ouliers, remove noise, normalize value)
- Deal with missing values and errors

## Features

Features are independent variables/columns. They impact the prediction or outcome of the analysis

### Feature Types :

- Categorical (Qualitative) :

- Numerical (Quantitative)

Categorical are divided into 2 types,

- Nominal : Named Categories
- Ordinal : Categories with an Implied Order

Also, there are 2 types of numericals,

- Discrete : Particular Numbers
- Continuous : Any Numeric Value

## Feature Selection

For selecting a feature, you need to be knowladgable about the domain. It needs to be related to the outcome.

## Feature Pruning :

Features are extracted from raw data, whether it be sturctured or unstructured. To remove unnecessary or redunadant feature is called pruning. On the other hand, selection is to hand pick or tailor the features. Usually pruning helps to reduce overfitting, improve generalization and speed up computation

### Why Feature Pruning is Important

1. **Reduces Dimensionality :** Too many features can lead to the curse of dimensionality, where the model struggles to learn patterns effectively.

2. **Improves Model Performance :** Eliminating irrelevant features prevents noise from affecting predictions.

3. **Enhances Interpretability :** A simpler model with fewer features is easier to interpret and analyze.

4. **Reduces Computational Cost :** Fewer features mean less memory usage and faster training times.

### Methods of Feature Pruning

1. **Filter Methods (Statistical Analysis)**

   - **Variance Thresholding** Removes features with low variance
   - **Correlation Analysis** Drops highly correlated features to remove redundancy.
   - **Mutual Information** Eliminates features that provide little information about the target variable.

2. **Wrapper Methods (Using Model Performance)**

   - **Recursive Feature Elimination (RFE):** Iteratively removes the least important feature and retrains the model.

   - **Feature Importance from Trees:** Decision trees and ensemble models (e.g., Random Forest, XGBoost) provide rankings of feature importance.

3. **Embedded Methods (Built-in Model Regularization)**

   - **LASSO (L1 Regularization):** Shrinks coefficients of irrelevant features to zero, effectively removing them.
   - **Ridge Regression (L2 Regularization):** Reduces feature impact rather than removing them entirely.

4. **Autoencoders and PCA (Dimensionality Reduction)**

   - **Principal Component Analysis (PCA):** Converts features into a lower-dimensional representation
   - **Autoencoders:** Neural networks that learn compressed representations of data.

## Feature Selection

Feature engineering is a crucial step in machine learning, as it helps models learn better by transforming raw data into meaningful features. There are three primary ways to obtain features: Feature Transformation, Feature Extraction, and Exploratory Data Analysis (EDA). Let’s break these down with real-world examples.

1. **Feature Transformations :**
   Feature transformation refers to modifying existing features to improve their usefulness for a machine learning model. This includes operations such as scaling, normalization, encoding categorical variables, and creating new features from existing ones.

   **Example: Scaling and Normalization**
   Imagine you are building a house price prediction model, and one of your features is the house size in square feet. The values range from 500 to 5000 square feet, but another feature, "number of bedrooms," ranges from 1 to 5. Since the scales of these two features are drastically different, the model might give undue importance to the larger numerical values.

   To solve this, you can normalize the values to fall between 0 and 1 or standardize them so they follow a normal distribution.


    **Example: Encoding Categorical Data**

    If you have a dataset of customers, a feature like "City" might contain values such as New York, Toronto, London, and Sydney. Since machine learning models work with numerical values, you need to convert these city names into numbers using techniques like one-hot encoding or label encoding.

1. **Feature Extraction :**
   Feature extraction involves deriving new, meaningful features from raw data. This is especially useful when working with text, images, and audio, where raw data alone is not sufficient.

   **Example: Text Data Processing (TF-IDF)**

   Suppose you're working on a spam detection model for emails. The raw text of emails is unstructured and cannot be directly used in a model. Feature extraction techniques like TF-IDF (Term Frequency-Inverse Document Frequency) can convert words into numerical values that represent their importance in an email. Words like "free" and "win" might have higher importance in spam emails compared to regular messages.

   **Example: Feature Extraction from Images**

   In a facial recognition system, raw pixel values of images are too detailed and noisy. Instead of using individual pixels, we can extract features such as edges, textures, and color histograms to identify key patterns that help distinguish different faces.

1. **Exploratory Data Analysis (EDA) :**

   EDA is the process of analyzing data to understand its characteristics before building a model. It helps in detecting missing values, outliers, data distributions, and relationships between features.

   **Example: Detecting Outliers in Salary Data**

   Suppose you are analyzing employee salaries in a company. Most employees earn between $30,000 and $100,000, but a few executives earn \$500,000+. These are outliers, and they might distort the model’s predictions. Using visualization techniques like histograms and box plots, you can identify such outliers and decide whether to remove them or adjust them.

   **Example: Checking Feature Correlation**

   Imagine you are building a model to predict a student's exam score based on several factors like study hours, school attendance, and number of extracurricular activities. If you find that study hours and exam scores have a strong correlation, you might prioritize this feature while ignoring unrelated ones (e.g., "favorite music genre").

### Feature Pruning and Selection on Our Dataset

As the selected dataset is on several different proteins and only 2 independent vairable exist (1 per method), there is no scope for pruning data/feature. And because of only 2 features in the given dataset, me and my teammate Jans have decided to collect extra features related to the protein compounds to further analyze the data
