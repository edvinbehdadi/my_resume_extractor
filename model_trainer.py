from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score

class ModelTrainer:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def train_models(self):
        features = list(self.dataframe.columns[:-1])
        X = self.dataframe[features]
        y = self.dataframe['TARGET']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "SVR": SVR(),
            "RandomForest": RandomForestRegressor(),
            "LinearRegression": LinearRegression(),
            "Lasso": Lasso()
        }

        best_model = None
        best_score = float('-inf')

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            print(f"{name} R2 Score: {score}")
            if score > best_score:
                best_score = score
                best_model = model

        print(f"Best Model: {best_model}")
        return best_model