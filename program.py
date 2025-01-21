import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

class MovieAnalyzer:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_importance = None
        
    def load_data(self, filepath):
        """Загрузка данных о фильмах/сериалах"""
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        return df
    
    def preprocess_data(self, df):
        required_columns = {'duration', 'year', 'rating', 'genre', 'country', 'director'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Отсутствуют необходимые колонки: {required_columns - set(df.columns)}")

        numeric_features = ['duration', 'year']
        categorical_features = ['genre', 'country', 'director']
        
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return self.preprocessor
    
    def train_model(self, df, target='rating'):
        df = df.dropna()
        if target not in df.columns:
            raise ValueError(f"Целевая переменная '{target}' отсутствует в данных")

        if df.shape[0] < 2:
            raise ValueError("Недостаточно данных для выполнения кросс-валидации. Требуется минимум 2 записи.")
        
        X = df.drop(target, axis=1)
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [None, 10]
        }
        
        cv_folds = max(2, min(3, df.shape[0] // 2))
        
        self.model = Pipeline([
            ('preprocessor', self.preprocess_data(df)),
            ('regressor', RandomForestRegressor(random_state=42))
        ])
        
        grid_search = GridSearchCV(self.model, param_grid, cv=cv_folds, scoring='r2', error_score='raise')
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        y_pred = self.model.predict(X_test)
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        joblib.dump(self.model, 'movie_rating_model.pkl')
        
        return metrics, (X_test, y_test, y_pred)
    
    def analyze_feature_importance(self, df, target='rating'):
        feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.named_steps['regressor'].feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance
    
    def create_visualizations(self, df):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        sns.boxplot(data=df, x='genre', y='rating', ax=axes[0, 0])
        axes[0, 0].set_title('Распределение рейтингов по жанрам')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        sns.scatterplot(data=df, x='year', y='rating', ax=axes[0, 1])
        axes[0, 1].set_title('Рейтинг фильмов по годам')
        
        df.groupby('year')['rating'].mean().plot(ax=axes[1, 0])
        axes[1, 0].set_title('Средний рейтинг по годам')
        
        sns.histplot(data=df, x='rating', bins=50, ax=axes[1, 1])
        axes[1, 1].set_title('Распределение рейтингов')
        
        plt.tight_layout()
        return fig
    
    def predict_rating(self, features):
        self.model = joblib.load('movie_rating_model.pkl')
        input_df = pd.DataFrame([features])
        prediction = self.model.predict(input_df)[0]
        return prediction

if __name__ == "__main__":
    analyzer = MovieAnalyzer()
    data = analyzer.load_data('movies.csv')
    metrics, _ = analyzer.train_model(data)
    print(f"R² score: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:,.0f}, MAPE: {metrics['mape']:.1f}%")
    feature_importance = analyzer.analyze_feature_importance(data)
    print(feature_importance)
    analyzer.create_visualizations(data)
    plt.show()
