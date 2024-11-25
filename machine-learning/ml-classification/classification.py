#1a
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/ML2024/train-val.csv')
df.head()

#2a
# Number of samples
num_samples = df.shape[0]

#Number of features
num_features = df.drop(columns=['RainTomorrow']).shape[1]

print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

#2b
#Dataframe containing only features
df_features = df

#Display data type and non-null count of each feature
feature_info = pd.DataFrame({
    'Data Type': df_features.dtypes,
    'Non-Null Count': df_features.notnull().sum(),
    'Missing Percentage': (df_features.isnull().sum()/len(df_features))*100
})

print("Feature data types and non-null counts:\n")
print(feature_info)

#Count the number of features of each data type
dtype_counts = df_features.dtypes.value_counts()
print("\nNumber of Features by Data Type:\n")
print(dtype_counts)

#2c
#Display the feature labels (column names)
print("Feature Labels:\n")
print(df_features.columns.tolist())

#2d
#Count the number of unique categories in the clas label 'RainTomorrow'
num_categories = df['RainTomorrow'].nunique()
print(f"Number of categories in 'RainTomorrow': {num_categories}")

#2e
#Count the number of samples in each class of 'RainTomorrow'
class_counts = df['RainTomorrow'].value_counts()
print("Number of samples in each class of 'RainTomorrow':\n")
print(class_counts)

#Calculate the imbalance ratio
imbalance_ratio = class_counts[1]/class_counts[0]
print(f"\nImbalance Ratio (Rain : No Rain): {imbalance_ratio:.2f}")

#2f
#Create a Dataframe with only numerical features
num_features_df = df_features.select_dtypes(include=['int64', 'float64'])

#Calculate the correlation matrix for numerical features in df_features
correlation_matrix = num_features_df.corr()

#Display the correlation matrix
print("Correlation Matrix:\n")
print(correlation_matrix)

# Calculate correlation between each feature in `num_features_df` and `RainTomorrow` from `df`
correlation_with_target = num_features_df.corrwith(df['RainTomorrow'])

# Display the 1xfeatures matrix
print("\nCorrelation of each numerical feature with RainTomorrow:\n", correlation_with_target)

import matplotlib.pyplot as plt
import seaborn as sns
#Plot a heatmap of the correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot =True, cmap='coolwarm',fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

#2g
numerical_descriptive_stats = num_features_df.describe()
print("Numerical Descriptive Statistics:\n")
print(numerical_descriptive_stats)
#for categorical features
categorical_columns = df_features.select_dtypes(include=['object']).columns
for col in categorical_columns:
  print(f"\nFrequency distribution for {col} :\n",df_features[col].value_counts(normalize=True)*100)

#3abcd
# Make a copy of the original DataFrame
df_selected = df.copy()

#3βγδ
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# Feature engineering for numerical columns
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns if isinstance(X, pd.DataFrame) else None
        return self

    def transform(self, X):
        # Ensure the input is a DataFrame
        X = pd.DataFrame(X, columns=self.feature_names_in_) if not isinstance(X, pd.DataFrame) else X.copy()

        # Create engineered features
        X['TempRange'] = X['MaxTemp'] - X['MinTemp']
        X['PressureAvg'] = (X['Pressure9am'] + X['Pressure3pm']) / 2

        # Drop redundant columns
        columns_to_drop = ['MaxTemp', 'MinTemp', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Evaporation']
        X.drop(columns=columns_to_drop, errors='ignore', inplace=True)

        return X

# Function for encoding WindGustDir into sine and cosine
def wind_gust_dir_to_sin_cos(data):
    df = pd.DataFrame(data, columns=['WindGustDir'])
    direction_map = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90,
        'ESE': 112.5, 'SE': 135, 'SSE': 157.5, 'S': 180,
        'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270,
        'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    df['WindGustDirDegrees'] = df['WindGustDir'].map(direction_map)
    return pd.DataFrame({
        'WindGustDirSin': np.sin(np.deg2rad(df['WindGustDirDegrees'])),
        'WindGustDirCos': np.cos(np.deg2rad(df['WindGustDirDegrees']))
    })

# Function for calculating wind direction shifts and adding sine/cosine of WindDir9am
def wind_dir_to_shift(data):
    df = pd.DataFrame(data, columns=['WindDir9am', 'WindDir3pm'])
    direction_map = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90,
        'ESE': 112.5, 'SE': 135, 'SSE': 157.5, 'S': 180,
        'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270,
        'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    df['WindDir9amDegrees'] = df['WindDir9am'].map(direction_map)
    df['WindDir3pmDegrees'] = df['WindDir3pm'].map(direction_map)

    # Calculate wind shift
    df['WindShift'] = df['WindDir3pmDegrees'] - df['WindDir9amDegrees']
    df['WindShiftNormalized'] = (df['WindShift'] + 180) % 360 - 180


    # Calculate sine and cosine for WindDir9am
    df['WindDir9amSin'] = np.sin(np.deg2rad(df['WindDir9amDegrees']))
    df['WindDir9amCos'] = np.cos(np.deg2rad(df['WindDir9amDegrees']))

    return df[['WindShiftNormalized', 'WindDir9amSin', 'WindDir9amCos']]

# Function to extract month as sine and cosine
def date_to_month(data):
    data = pd.DataFrame(data, columns=['Date'])
    data['Date'] = pd.to_datetime(data['Date'])

    # Extract month and convert to sine/cosine
    month = data['Date'].dt.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    return pd.DataFrame({
        'MonthSin': month_sin,
        'MonthCos': month_cos
    })

# Pipelines for specific features
wind_shift_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('calculate_shift_and_sin_cos', FunctionTransformer(wind_dir_to_shift, validate=False)),
    ('scaler', MinMaxScaler(feature_range=(-1, 1)))
])

wind_gust_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encode_sin_cos', FunctionTransformer(wind_gust_dir_to_sin_cos, validate=False)),
    ('scaler', MinMaxScaler(feature_range=(-1, 1)))
])

date_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('month_features', FunctionTransformer(date_to_month, validate=False)),
    ('scaler', MinMaxScaler(feature_range=(-1, 1)))
])

location_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')),
    ('scaler', MinMaxScaler(feature_range=(-1, 1)))
])

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler(feature_range=(-1, 1)))  # Scale numerical features
])

numerical_features = [
    'ID', 'Rainfall', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
    'Humidity9am', 'Humidity3pm', 'Cloud9am', 'Cloud3pm', 'Temp3pm',
    'RainToday', 'TempRange', 'PressureAvg'
]

column_transformer = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('wind_shift', wind_shift_pipeline, ['WindDir9am', 'WindDir3pm']),
        ('location', location_pipeline, ['Location']),
        ('wind_gust', wind_gust_pipeline, ['WindGustDir']),
        ('date', date_pipeline, ['Date']),  # Use the new month-based pipeline
    ],
    remainder='passthrough'
)

preprocessor = Pipeline([
    ('feature_engineering', FeatureEngineer()),
    ('column_transformer', column_transformer)
])

#3e
X_processed = preprocessor.fit_transform(df_selected.drop(['RainTomorrow'], axis=1))

#3f
X = X_processed
y = df_selected['RainTomorrow']

#3g
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=43)

#4a
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

classifiers = {
    'Naive Bayes': GaussianNB(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(),
    'MLP': MLPClassifier(max_iter=400),
    'SVC': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

for name, classifier in classifiers.items():
    print(f"Training {name}...")
    classifier.fit(X_train, y_train)

#4b
predictions = {}
for name, classifier in classifiers.items():
    predictions[name] = classifier.predict(X_val) 
#4c
f1_scores = {}
for name, prediction in predictions.items():
    f1_scores[name] = f1_score(y_val, prediction)
    print(f"F1 Score for {name}: {f1_scores[name]}")
#4d
plt.figure(figsize=(10, 6))
plt.bar(f1_scores.keys(), f1_scores.values())
plt.title('Classifier Performance (F1 Score)')
plt.xlabel('Classifier')
plt.ylabel('F1 Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show() # Execute the code yourself to see the output.
best_classifier_name = max(f1_scores, key=f1_scores.get)
print(f"\nThe best performing classifier is: {best_classifier_name} with an F1 score of {f1_scores[best_classifier_name]}")

#5a
test_df = pd.read_csv('/content/drive/MyDrive/ML2024/test.csv')

X_test_processed = preprocessor.transform(test_df.drop(['RainTomorrow'], axis=1, errors='ignore'))  #
best_classifier = classifiers[best_classifier_name]
predictions_test = best_classifier.predict(X_test_processed)

#5b
submission_df = pd.DataFrame({'ID': test_df['ID'], 'RainTomorrow': predictions_test})
print(submission_df.head())
# Save the DataFrame to a CSV file
submission_df.to_csv('submission1.csv', index=False)

#6a
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Adjusted parameter grids for each classifier
param_grids = {
    'Naive Bayes': {
        # No hyperparameters for GaussianNB, using default settings
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7],  # Reduced range of neighbors
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'LogisticRegression': {
        'penalty': ['l2'],                # Focus on l2 regularization
        'C': [0.1, 1, 10],                # Generalized range for regularization
        'solver': ['liblinear'],          # Compatible with l2
        'max_iter': [100, 500]            # Sufficient for convergence
    },
    'MLP': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Fewer configurations
        'activation': ['tanh'],                           # Focused on tanh
        'solver': ['adam'],                               # Reliable solver
        'alpha': [0.0001, 0.001],                         # Regularization strength
        'learning_rate': ['constant'],                    # Fixed learning rate
        'max_iter': [400]                                 # Standard iteration limit
    },
    'SVC': {
        'C': [0.1, 1, 10],           # General range for C
        'kernel': ['linear', 'rbf'], # Limited to common kernels
        'gamma': ['scale', 'auto']   # Standard gamma options
    },
    'Decision Tree': {
        'criterion': ['gini'],             # Removed log_loss for simplicity
        'max_depth': [10, 20, None],       # Reasonable depth options
        'min_samples_split': [2, 10]       # Common split thresholds
    },
    'Random Forest': {
        'n_estimators': [50, 100],         # Common numbers of estimators
        'criterion': ['gini'],
        'max_depth': [10, 20, None],       # Standard depth range
        'min_samples_split': [2, 10]       # Usual splits
    }
}

# Dictionary to store the best estimators
best_estimators = {}

# Loop through each classifier and perform GridSearchCV
for name, classifier in classifiers.items():
    print(f"Tuning {name}...")
    if name not in param_grids or not param_grids[name]:
        print(f"No hyperparameters to tune for {name}, using default settings.")
        best_estimators[name] = classifier.fit(X_train, y_train)
        continue

    grid_search = GridSearchCV(
        classifier,
        param_grids[name],
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1  # Optional: Set to 2 for more detailed progress output
    )

    # Fit the grid search on the training data
    grid_search.fit(X_train, y_train)

    # Store the best estimator and print the best parameters and F1 score
    best_estimators[name] = grid_search.best_estimator_
    print(f"Best F1 Score for {name}: {grid_search.best_score_}")
    print(f"Best Parameters for {name}: {grid_search.best_params_}")

# Example: Access the best estimator for LogisticRegression
print("\nBest LogisticRegression Estimator:", best_estimators['LogisticRegression'])

#6a
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Store the best classifiers with their hyperparameters
best_classifiers = {
    'Naive Bayes': GaussianNB(),  # No hyperparameters to tune
    'KNeighborsClassifier': KNeighborsClassifier(
        n_neighbors=3,
        weights='distance',
        metric='manhattan'
    ),
    'LogisticRegression': LogisticRegression(
        C=10,
        penalty='l2',
        solver='liblinear',
        max_iter=100
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(50,),
        activation='tanh',
        solver='adam',
        alpha=0.001,
        learning_rate='constant',
        max_iter=200,
        random_state=123
    ),
    'SVC': SVC(
        C=10,
        kernel='rbf',
        gamma='scale'
    ),
    'Decision Tree': DecisionTreeClassifier(
        criterion='gini',
        max_depth=10,
        min_samples_split=2
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=50,
        criterion='gini',
        max_depth=20,
        min_samples_split=2
    )
}

#6b
# Fit all classifiers on the training data
for name, classifier in best_classifiers.items():
    print(f"Fitting {name}...")
    classifier.fit(X_train, y_train)

# Make predictions on the validation data
predictions = {}
for name, classifier in best_classifiers.items():
    predictions[name] = classifier.predict(X_val)

# Example: Display predictions for one classifier
print("Predictions for LogisticRegression:", predictions['LogisticRegression'])

#6c
f1_scores = {}
for name, prediction in predictions.items():
    f1_scores[name] = f1_score(y_val, prediction)
    print(f"F1 Score for {name}: {f1_scores[name]}")
#6d
plt.figure(figsize=(10, 6))
plt.bar(f1_scores.keys(), f1_scores.values())
plt.title('Classifier Performance after Hyperparameter Tuning (F1 Score)')
plt.xlabel('Classifier')
plt.ylabel('F1 Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show() # Execute the code yourself to see the output.best_classifier_name = max(f1_scores, key=f1_scores.get)

best_classifier_name_improved = max(f1_scores, key=f1_scores.get)
print(f"\nThe best performing classifier is: {best_classifier_name_improved} with an F1 score of {f1_scores[best_classifier_name_improved]}")

#7a
X_test_processed = preprocessor.transform(test_df.drop(['RainTomorrow'], axis=1, errors='ignore'))  #
best_classifier_improved = best_classifiers[best_classifier_name_improved]
predictions_test_improved = best_classifier_improved.predict(X_test_processed)

#7b
submission_df_improved = pd.DataFrame({'ID': test_df['ID'], 'RainTomorrow': predictions_test_improved})
print(submission_df_improved.head())
# Save the DataFrame to a CSV file
submission_df_improved.to_csv('submission_improved1.csv', index=False)