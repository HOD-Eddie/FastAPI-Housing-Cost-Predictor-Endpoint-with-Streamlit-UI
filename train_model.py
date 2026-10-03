import joblib
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Changed from LogisticRegression

# Ensure target directory framework exists
target_dir = Path("app/models")
target_dir.mkdir(parents=True, exist_ok=True)

# Fetch and unpack the California Housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split the data (80% training, 20% validation testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# Initialize and fit the REGRESSION model
# n_estimators=100 sets up 100 decision trees to average out the price prediction
model = RandomForestRegressor(n_estimators=100, random_state=44)
model.fit(X_train, y_train)

# Evaluate the model performance on the test set using R-squared score
score = model.score(X_test, y_test)
print(f"--> Model training complete. R² Accuracy Score: {score:.4f}")

# Dump the trained model file cleanly to disk
model_filename = target_dir / "housing_model.joblib"
joblib.dump(model, model_filename)
print(f"--> Success! Model saved safely to: {model_filename}")


## the code instructions below are useful when I import a new dataset,
# the allow me the see how data must be structured or defined in my pydantic schemas

# Inspect the feature column names => column names are re-written in nice snake_case
# filed names that works well with Python, where applicable.
# for this use case, MedInc becomes median_income
# print("Features expected by this dataset:")
# print(housing.feature_names)


# Inspect what a single row of input looks like
# print("\nExample data row:")
# print(housing.data[0])

# Inspect what the output target represents
# print("\nTarget variable meaning:")
# print(housing.target_names)

# the entire process recaptured
        # DESTINATION_DIR = Path("the directory")
        # DESTINATION_DIR.mkdir(parents=True, exist_ok=True)

        # instance = imported_dataset()

        # X = instance.data
        # y = instance.target

        # X_train, X_test, y_train, y_test = data_split_func(X, y, test_size = "some size (int)", random_state="some value (int)")

        # model = SomeModelClassNameHere(n_estimators=100, random_state=44)
        # model.fit(X_train, y_train)

        # score = model.score(X_test, y_test)
        # print(f"--> Model training complete. R² Accuracy Score: {score:.4f}")

        # model_filename = DESTINATION_DIR / "model_name.joblib"
        # joblib.dump(model, model_filename)
        # print(f"--> Success! Model saved safely to: {model_filename}")