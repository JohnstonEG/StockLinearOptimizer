import streamlit as st
import pandas as pd
import statsmodels.api as sm
import itertools as it 
from statsmodels.stats.outliers_influence import variance_inflation_factor

class StockAnalyzer:
    def __init__(self, df):
        self.df = df
        self.dfx = None
        self.dfy = None
        self.best_model = None
        self.best_features = None
        self.best_score = None
        self.feature_categories = None
    
    def CleanData(self, drop_column_threshold=0.1):
        df = self.df.copy()
        #remove commas from data if present
        df = df.replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce')
        max_na = len(df) * drop_column_threshold
        df = df.drop(columns=df.columns[df.isnull().sum() > max_na])
        df = df.dropna()
        if df.empty:
            return None, None
        self.dfx = df.drop(columns=df.columns[0])
        self.dfy = df.iloc[:, 0]
        return self.dfx, self.dfy

    def classify_features(self):
        if self.dfx is None:
            return None
        #defining keywords: will determine optimal way later if feasible 
        risk_keywords = ['risk', 'debt', 'beta', 'volatility', 'leverage']
        profitability_keywords = ['Return', 'profit', 'margin', 'nim', 'income', 'earnings', 'eps', 'roe', 'roa']
        growth_keywords = ['Rate', 'growth', 'change', 'increase', '%', 'yoy', 'chg']
        self.feature_categories = {'risk': [], 'profitability': [], 'growth': [], 'other': []}
        for column in self.dfx.columns:
            col_lower = column.lower()
            if any(keyword in col_lower for keyword in risk_keywords):
                self.feature_categories['risk'].append(column)
            elif any(keyword in col_lower for keyword in profitability_keywords):
                self.feature_categories['profitability'].append(column)
            elif any(keyword in col_lower for keyword in growth_keywords):
                self.feature_categories['growth'].append(column)
            else:
                self.feature_categories['other'].append(column)
        return self.feature_categories
    
    def validate_feature_combination(self, features, strict_requirements=True):
        #requires at least one of each feature when strict
        if self.feature_categories is None:
            raise ValueError("Features have not been classified. Run classify_features() first.")
        
        if not strict_requirements:
            return True  # Accept any combination when not strict
            
        has_risk = any(feature in self.feature_categories['risk'] for feature in features)
        has_profitability = any(feature in self.feature_categories['profitability'] for feature in features)
        has_growth = any(feature in self.feature_categories['growth'] for feature in features)
        
        return has_risk and has_profitability and has_growth
    
    def remove_multicollinear_features(self, threshold=5.0):
        #uses VIF to remove weak features that are highly multicollinear  
        X = sm.add_constant(self.dfx)
        dropped_features = []
        while True:
            vif_data = pd.DataFrame({'Feature': X.columns, 'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
            high_vif = vif_data[vif_data["VIF"] > threshold]
            if high_vif.empty:
                break
            feature_to_drop = high_vif.sort_values("VIF", ascending=False).iloc[0]["Feature"]
            if feature_to_drop == "const":
                break
            X = X.drop(columns=[feature_to_drop])
            dropped_features.append(feature_to_drop)
        self.dfx = X.drop(columns="const", errors="ignore")
        return dropped_features

    def find_best_model(self, criterion='adj_r2', p_value=0.10, max_features=None, strict_requirements=True):
        if self.dfx is None or self.dfy is None:
            return None
            
        X = sm.add_constant(self.dfx)
        feature_names = list(X.columns[1:])
        current_features = []
        best_score = float('-inf') if criterion == 'adj_r2' else float('inf')
        
        if max_features is None:
            max_features = len(feature_names)
        max_features = min(max_features, len(feature_names))

        #limits model search to max features. Faster with less features chosen
        while len(current_features) < max_features:
            best_new_feature = None
            best_new_score = best_score
            
            for feature in feature_names:
                if feature in current_features:
                    continue
                    
                test_features = current_features + [feature]
                
                # Only apply category validation if strict requirements are enabled
                if (strict_requirements and 
                    len(test_features) >= 3 and 
                    not self.validate_feature_combination(test_features, strict_requirements)):
                    continue
                    
                X_subset = X[['const'] + test_features]
                model = sm.OLS(self.dfy, X_subset.astype(float)).fit()
                
                if any(model.pvalues[1:] >= p_value):
                    continue

                #scores based on a metric. Need to include option to choose
                score = (model.rsquared_adj if criterion == 'adj_r2' 
                        else model.aic if criterion == 'aic' 
                        else model.bic)
                
                if ((criterion == 'adj_r2' and score > best_new_score) or 
                    (criterion in ['aic', 'bic'] and score < best_new_score)):
                    best_new_score = score
                    best_new_feature = feature
                    
            if best_new_feature is None:
                break
                
            current_features.append(best_new_feature)
            best_score = best_new_score
            
            X_final = X[['const'] + current_features]
            best_model = sm.OLS(self.dfy, X_final.astype(float)).fit()
            
        # Remove minimum feature requirement when not strict
        self.best_model = best_model if (not strict_requirements or len(current_features) >= 3) else None
        self.best_features = current_features
        self.best_score = best_score
        
        return self.best_model

    def add_performance_metrics(self):
        if self.best_model is None:
            return None
            
        # Calculate metrics
        predictions = self.best_model.predict(sm.add_constant(self.dfx[list(self.best_features)]))
        residuals = self.dfy - predictions
        
        metrics = {
            'R-squared': self.best_model.rsquared,
            'Adjusted R-squared': self.best_model.rsquared_adj,
            'AIC': self.best_model.aic,
            'BIC': self.best_model.bic,
            'Mean Absolute Error': abs(residuals).mean(),
            'Root Mean Squared Error': (residuals ** 2).mean() ** 0.5,
            'Number of Features': len(self.best_features)
        }
        
        return metrics

st.title("Stock Analyzer App")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
p_value = st.slider("Select p-value threshold", 0.001, 0.999, step=0.001)
drop_threshold = st.slider("Max % of column NA before drop", 0.001, 0.999, step=0.01)
max_features = st.slider("Maximum number of features", 3, 15, 10)
strict_requirements = st.checkbox("Strict Category Requirements", value=True, 
    help="If checked, requires at least one feature from each category (Risk, Profitability, Growth)")
    
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    analyzer = StockAnalyzer(df)
    dfx, dfy = analyzer.CleanData(drop_threshold)
    if dfx is not None:
        feature_categories = analyzer.classify_features()
        dropped_features = analyzer.remove_multicollinear_features()
        best_model = analyzer.find_best_model(p_value=p_value, max_features=max_features, strict_requirements=strict_requirements)
        st.write("### Feature Categories", feature_categories)
        st.write("### Removed Multicollinear Features", dropped_features)
        if best_model:
            st.write("### Best Model Summary")
            st.text(best_model.summary())
        else:
            st.write("No valid model found. Try adjusting the p-value or increasing max features")
    else:
        st.write("No valid data found. Try adjusting the drop threshold")
