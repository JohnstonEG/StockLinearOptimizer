import streamlit as st
import pandas as pd
import statsmodels.api as sm
import itertools
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
    
    def validate_feature_combination(self, features):
        if self.feature_categories is None:
            raise ValueError("Features have not been classified. Run classify_features() first.")
            
        has_risk = any(feature in self.feature_categories['risk'] for feature in features)
        has_profitability = any(feature in self.feature_categories['profitability'] for feature in features)
        has_growth = any(feature in self.feature_categories['growth'] for feature in features)
        
        return has_risk and has_profitability and has_growth
    
    def remove_multicollinear_features(self, threshold=5.0):
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

    def find_best_model(self, criterion='adj_r2', p_value=0.10):
        if self.dfx is None or self.dfy is None:
            return None
        X = sm.add_constant(self.dfx)
        feature_names = X.columns[1:]
        best_model, best_features, best_score = None, None, float('-inf') if criterion == 'adj_r2' else float('inf')
        for r in range(3, len(feature_names) + 1):
            for subset in itertools.combinations(feature_names, r):
                X_subset = X[['const'] + list(subset)]
                model = sm.OLS(self.dfy, X_subset.astype(float)).fit()
                if all(model.pvalues[1:] < p_value):  
                    score = model.rsquared_adj if criterion == 'adj_r2' else model.aic if criterion == 'aic' else model.bic
                    if (criterion == 'adj_r2' and score > best_score) or (criterion in ['aic', 'bic'] and score < best_score):
                        best_model, best_features, best_score = model, subset, score
        self.best_model, self.best_features, self.best_score = best_model, best_features, best_score
        return best_model

st.title("Stock Analyzer App")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
p_value = st.slider("Select p-value threshold", 0.001, 0.5, 0.10)
drop_threshold = st.slider("Max % of column NA before drop", 0.01, 0.5, 0.1)
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    analyzer = StockAnalyzer(df)
    dfx, dfy = analyzer.CleanData(drop_threshold)
    if dfx is not None:
        feature_categories = analyzer.classify_features()
        dropped_features = analyzer.remove_multicollinear_features()
        best_model = analyzer.find_best_model(p_value=p_value)
        st.write("### Feature Categories", feature_categories)
        st.write("### Removed Multicollinear Features", dropped_features)
        if best_model:
            st.write("### Best Model Summary")
            st.text(best_model.summary())
        else:
            st.write("No valid model found.")
