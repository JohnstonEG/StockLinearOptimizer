# %%
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import itertools

# %%
class StockAnalyzer:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.dfx = None
        self.dfy = None
        self.best_model = None  # Store best regression model
        self.best_features = None
        self.best_score = None
        
    def CleanData(self, drop_column_threshold = 0.1):
        #define filepath
        df = self.df.copy()
        
        # Remove commas and convert to float
        df = df.replace(',', '', regex=True)  # Remove thousand separators
        df = df.apply(pd.to_numeric, errors='coerce')  # Convert everything to float

        #remove columns with excess missing inputs
        #max_na based on threshold
        max_na = len(df) * drop_column_threshold
        
        #Count na in column
        column_nan_count = df.isnull().sum()
        
        #drop columns with na count > max_na
        df = df.drop(columns=column_nan_count[column_nan_count > max_na].index)
        
        #drop rows with na 
        df = df.dropna()
        
        if df.empty:
            print("Warning: DataFrame is empty after cleaning!")
            self.dfx, self.dfy = None, None
            return None, None
        
        #define variable columns
        self.dfx = df.drop(columns=df.columns[0])
        self.dfy = df.iloc[: , 0]
        
        return self.dfx, self.dfy  
            
    def classify_features(self):
        """Classify features into risk, profitability, growth, and other categories based on keywords."""
        if self.dfx is None:
            raise ValueError("Data has not been cleaned. Run CleanData() first.")

        # Define keywords for each category
        risk_keywords = ['risk', 'debt', 'beta', 'volatility', 'leverage']
        profitability_keywords = ['profit', 'margin', 'nim', 'income', 'earnings', 'eps', 'roe', 'roa']
        growth_keywords = ['growth', 'change', 'increase', '%', 'yoy', 'chg']

        # Initialize category dictionaries
        self.feature_categories = {
            'risk': [],
            'profitability': [],
            'growth': [],
            'other': []
        }

        # Classify each feature
        for column in self.dfx.columns:
            col_lower = column.lower()

            # Check each category
            if any(keyword in col_lower for keyword in risk_keywords):
                self.feature_categories['risk'].append(column)
            elif any(keyword in col_lower for keyword in profitability_keywords):
                self.feature_categories['profitability'].append(column)
            elif any(keyword in col_lower for keyword in growth_keywords):
                self.feature_categories['growth'].append(column)
            else:
                self.feature_categories['other'].append(column)

        # Print classification results
        print("\nFeature Classification:")
        for category, features in self.feature_categories.items():
            print(f"\n{category.capitalize()} features:")
            print(features if features else "None found")
            
    def validate_feature_combination(self, features):
        """Check if feature combination includes at least one from each required category."""
        if self.feature_categories is None:
            raise ValueError("Features have not been classified. Run classify_features() first.")
            
        has_risk = any(feature in self.feature_categories['risk'] for feature in features)
        has_profitability = any(feature in self.feature_categories['profitability'] for feature in features)
        has_growth = any(feature in self.feature_categories['growth'] for feature in features)
        
        return has_risk and has_profitability and has_growth
    
    def remove_multicollinear_features(self, threshold=5.0):
        X = sm.add_constant(self.dfx)  # Add intercept
        dropped_features = []
        
        #vif framework to remove weak collinear varaibles
        while True:
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

            # Drop feature with highest VIF (if above threshold)
            high_vif = vif_data[vif_data["VIF"] > threshold]
            if high_vif.empty:
                break  # Stop if no high-VIF features remain

            feature_to_drop = high_vif.sort_values("VIF", ascending=False).iloc[0]["Feature"]
            if feature_to_drop == "const":
                break  # Never drop intercept

            X = X.drop(columns=[feature_to_drop])
            dropped_features.append(feature_to_drop)

        print(f"Removed multicollinear features: {dropped_features}")
        self.dfx = X.drop(columns="const", errors="ignore")  # Update cleaned X
    
    def find_best_model(self, criterion='adj_r2', p_value=0.10):
        if self.dfx is None or self.dfy is None:
            raise ValueError("Data has not been cleaned. Run CleanData() first.")
        if self.feature_categories is None:
            raise ValueError("Features have not been classified. Run classify_features() first.")

        
        X = sm.add_constant(self.dfx)  # Add intercept
        feature_names = X.columns[1:]  # Exclude intercept in naming
        best_model = None
        best_features = None
        
        #ensures at least one model is selected 
        best_score = float('-inf') if criterion == 'adj_r2' else float('inf')

        # Try all possible feature combinations
        for r in range(3, len(feature_names) + 1):
            for subset in itertools.combinations(feature_names, r):
                # Skip combinations that don't meet category requirements
                if not self.validate_feature_combination(subset):
                    continue
                X_subset = X[['const'] + list(subset)]

                model = sm.OLS(self.dfy, X_subset.astype(float)).fit()
                p_values = model.pvalues[1:]  # Exclude intercept p-value

                # Check if all selected features meet the p-value criterion
                if all(p_values < p_value):  
                    score = (
                        model.rsquared_adj if criterion == 'adj_r2' else
                        model.aic if criterion == 'aic' else
                        model.bic
                    )

                    # Update best model based on criterion
                    if (criterion == 'adj_r2' and score > best_score) or \
                       (criterion in ['aic', 'bic'] and score < best_score):
                        best_model = model
                        best_features = subset
                        best_score = score

        # Store best model & features
        self.best_model = best_model
        self.best_features = best_features
        self.best_score = best_score
        
        if best_model is None:
            print("No valid model found meeting all criteria (p-value and category requirements)")
        else:
            print("\nBest Model Features by Category:")
            for category in ['risk', 'profitability', 'growth', 'other']:
                category_features = [f for f in best_features if f in self.feature_categories[category]]
                print(f"\n{category.capitalize()}:")
                print(category_features if category_features else "None")

    def get_best_model_summary(self):
        """Returns the summary of the best regression model."""
        if self.best_model:
            return self.best_model.summary()
        else:
            return "No model found. Run find_best_model() first."

# %%
analyzer = StockAnalyzer("C:/Users/slick/Downloads/Relative Eval.csv")
#defines what columns to drop if their NaNs are greater than X% of total rows/companies
analyzer.CleanData(drop_column_threshold = 0.10)
analyzer.classify_features()
analyzer.remove_multicollinear_features(threshold = 3.0)
analyzer.find_best_model(criterion='adj_r2', p_value = .10)
analyzer.get_best_model_summary()

# %%



