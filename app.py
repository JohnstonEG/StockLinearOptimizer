import streamlit as st
import pandas as pd
import statsmodels.api as sm
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

        # Initialize session state for feature categories if not exists
        if 'feature_categories' not in st.session_state:
            st.session_state.feature_categories = None
    
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
        if st.session_state.feature_categories is None:
            risk_keywords = ['risk', 'debt', 'beta', 'volatility', 'leverage']
            profitability_keywords = ['Return', 'profit', 'margin', 'nim', 'income', 'earnings', 'eps', 'roe', 'roa']
            growth_keywords = ['Rate', 'growth', 'change', 'increase', '%', 'yoy', 'chg']
            
            categories = {'risk': [], 'profitability': [], 'growth': [], 'other': [], 'discard': []}
            for column in self.dfx.columns:
                col_lower = column.lower()
                if any(keyword in col_lower for keyword in risk_keywords):
                    categories['risk'].append(column)
                elif any(keyword in col_lower for keyword in profitability_keywords):
                    categories['profitability'].append(column)
                elif any(keyword in col_lower for keyword in growth_keywords):
                    categories['growth'].append(column)
                else:
                    categories['other'].append(column)
            
            st.session_state.feature_categories = categories
        else:
            # Ensure the 'discard' key is present
            if 'discard' not in st.session_state.feature_categories:
                st.session_state.feature_categories['discard'] = []
                
        self.feature_categories = st.session_state.feature_categories
        return self.feature_categories
    
    def validate_feature_combination(self, features, strict_requirements=True):
        if self.feature_categories is None:
            raise ValueError("Features have not been classified. Run classify_features() first.")
        
        if not strict_requirements:
            return True  # Accept any combination when not strict
            
        has_risk = any(feature in self.feature_categories['risk'] for feature in features)
        has_profitability = any(feature in self.feature_categories['profitability'] for feature in features)
        has_growth = any(feature in self.feature_categories['growth'] for feature in features)
        
        return has_risk and has_profitability and has_growth
    
    def remove_multicollinear_features(self, threshold=5.0):
        valid_features = [f for f in self.dfx.columns if f not in self.feature_categories['discard']]
        X = sm.add_constant(self.dfx[valid_features])
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
            
        valid_features = [f for f in self.dfx.columns if f not in self.feature_categories['discard']]
        X = sm.add_constant(self.dfx[valid_features])
        feature_names = list(X.columns[1:])
        current_features = []
        best_score = float('-inf') if criterion == 'adj_r2' else float('inf')

        if max_features is None:
            max_features = len(feature_names)
        max_features = min(max_features, len(feature_names))
        
        while len(current_features) < max_features:
            best_new_feature = None
            best_new_score = best_score
            
            for feature in feature_names:
                if feature in current_features:
                    continue
                    
                test_features = current_features + [feature]
                
                if (strict_requirements and 
                    len(test_features) >= 3 and 
                    not self.validate_feature_combination(test_features, strict_requirements)):
                    continue
                    
                X_subset = X[['const'] + test_features]
                model = sm.OLS(self.dfy, X_subset.astype(float)).fit()
                
                if any(model.pvalues[1:] >= p_value):
                    continue
                    
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

st.title("Stock Linear Regression Analyzer")
st.markdown("""
## **Project Goal**
My goal with this project is to create a tool that helps users filter and test variables they consider most relevant in their stock analysis.  
By selecting appropriate financial metrics, users can analyze how different factors—such as growth, profitability, and risk—affect a valuation measure like **P/E ratio, P/B ratio, or another dependent variable**.

## **How It Works**
1. **Upload a CSV file** with stock-related data.
2. **Select a valuation metric** (dependent variable) such as P/E ratio.
3. **Filter financial indicators** (independent variables/features) across **growth, profitability, and risk** categories.
4. **Adjust filtering settings** (p-value threshold, NA column drop percentage).
5. **Identify the best model**, removing multicollinearity and selecting variables with statistical significance.

---
""")

def render_feature_management(categories):
    st.write("### Manage Feature Categories")
    st.markdown("Select a new category for any feature you want to reclassify.")

    cols = st.columns([1, 1.5, 1, 1, 1])  # Adjusted for the new "discard" category
    new_categories = {
        'risk': categories['risk'].copy(),
        'profitability': categories['profitability'].copy(),
        'growth': categories['growth'].copy(),
        'other': categories['other'].copy(),
        'discard': categories['discard'].copy()
    }

    for idx, (category, features) in enumerate(categories.items()):
        with cols[idx]:
            st.subheader(category.title())
            for feature in features:
                target_category = st.selectbox(
                    f"Move {feature}",
                    options=['Keep Here'] + [cat for cat in categories.keys() if cat != category],
                    key=f"{category}_{feature}"
                )
                if target_category != 'Keep Here':
                    if feature in new_categories[category]:
                        new_categories[category].remove(feature)
                        new_categories[target_category].append(feature)

    if st.button("Update Categories"):
        st.session_state.feature_categories = new_categories
        st.rerun()
        
uploaded_file = st.file_uploader("Upload CSV", type=["csv"],
    help="Upload a CSV file in the format: column 1 = dependent variable, columns 2+ = independent variables")
p_value = st.slider("Select p-value threshold", 0.001, 0.999, step=0.001,
    help="Maximum p-value acceptable for feature selection")
drop_threshold = st.slider("Max % of column NA before drop", 0.001, 0.999, step=0.01,
    help="Maximum percentage of NA values in a column before dropping the column")
max_features = st.slider("Maximum number of features", 3, 15, 10,
    help="Maximum number of features to include in the model")
strict_requirements = st.checkbox("Strict Category Requirements", value=True, 
    help="If checked, requires at least one feature from each category (Risk, Profitability, Growth)")
    
if uploaded_file:
    if ('uploaded_file_name' not in st.session_state or 
        st.session_state.uploaded_file_name != uploaded_file.name):
        st.session_state.feature_categories = None  # Reset only if the file is new
        st.session_state.uploaded_file_name = uploaded_file.name

    df = pd.read_csv(uploaded_file)
    analyzer = StockAnalyzer(df)
    dfx, dfy = analyzer.CleanData(drop_threshold)
    if dfx is not None:
        feature_categories = analyzer.classify_features()
        
        # Render feature management UI
        render_feature_management(feature_categories)
        
        dropped_features = analyzer.remove_multicollinear_features()
        best_model = analyzer.find_best_model(p_value=p_value, max_features=max_features, strict_requirements=strict_requirements)
        
        # Show dropped features and model results
        st.write("### Removed Multicollinear Features")
        st.write(dropped_features)
        
        if best_model:

            # Get performance metrics as a dictionary
            metrics = analyzer.add_performance_metrics()
            
            # Convert the dictionary to a DataFrame for display
            metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
            
            # Display the performance metrics table
            st.write("### Model Performance Metrics")
            st.table(metrics_df)
            st.write("### Best Model Coefficients")
            # Use summary2() to get a cleaner table representation
            summary2 = best_model.summary2()
            # The coefficients table is typically the second table in summary2()
            coef_table = summary2.tables[1]
            st.dataframe(coef_table)
        else:
            st.write("No valid model found. Try adjusting the p-value or increasing max features")
    else:
        st.write("No valid data found. Try adjusting the drop threshold")