import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt
import plotly.express as px

# Set full screen and page config
st.set_page_config(page_title="ANN Classification Dashboard", layout="wide")

# Custom CSS for beautiful UI
st.markdown('''
    <style>
    body, .stApp {background: linear-gradient(135deg, #f8f6f0 0%, #f3e9dc 100%) !important;}
    .block-container {padding-top: 2rem;}
    .sidebar .sidebar-content {background: #f8f6f0; color: #333;}
    .css-1d391kg {background: #f8f6f0 !important;}
    .stButton>button, .stRadio>div>label {font-size: 1.1rem; border-radius: 8px; margin-bottom: 0.5rem;}
    .stRadio>div>label {background: #f9f6f2; color: #333; padding: 0.5rem 1rem; border-radius: 8px; transition: 0.2s; border: 1px solid #e6dfc8;}
    .stRadio>div>label:hover {background: #f3e9dc; color: #4e54c8;}
    .stRadio>div>label[data-selected="true"] {background: #f3e9dc; color: #4e54c8; border: 1.5px solid #4e54c8;}
    .stDataFrame, .stPlotlyChart, .stBarChart {background: #fff; border-radius: 12px; box-shadow: 0 2px 16px rgba(0,0,0,0.07); margin-bottom: 2rem; border: 1px solid #f3e9dc;}
    h2, h3, h4 {color: #4e54c8; font-family: 'Montserrat', sans-serif;}
    .stMarkdown {font-size: 1.1rem; color: #333;}
    </style>
''', unsafe_allow_html=True)

# Define EA dataset columns
numeric_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']

drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']

# Define Iris dataset columns
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Utility: train baseline and FE models and cache results
@st.cache_data
def load_and_train(df, target_col, feature_cols, encode_cats=None, fe_drop=None):
    # Encode target
    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col].astype(str))
    # Baseline: encode categoricals, no scaling
    present_features_base = [col for col in feature_cols if col in df.columns]
    X = df[present_features_base].copy()
    if encode_cats:
        for c in encode_cats:
            if c in X.columns:
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )
    # Baseline model
    num_classes = len(np.unique(y_train))
    if num_classes == 2:
        baseline = models.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        loss = 'binary_crossentropy'
    else:
        baseline = models.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        loss = 'sparse_categorical_crossentropy'
    baseline.compile('adam', loss=loss, metrics=['accuracy'])
    h_base = baseline.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=16, verbose=0)
    y_pred = np.argmax(baseline.predict(X_test), axis=1) if num_classes>2 else (baseline.predict(X_test)>0.5).astype(int).flatten()
    cm_base = confusion_matrix(y_test, y_pred)
    acc_base = accuracy_score(y_test, y_pred)
    # Feature engineering: drop, scale, PCA
    df_fe = df.copy()
    if fe_drop: df_fe.drop(columns=fe_drop, inplace=True)
    # Only select features that are present after dropping
    present_features = [col for col in feature_cols if col in df_fe.columns]
    Xf = df_fe[present_features].copy()
    # Only encode categorical columns that are present
    for c in (encode_cats or []):
        if c in Xf.columns:
            Xf[c] = LabelEncoder().fit_transform(Xf[c].astype(str))
    yfe = df_fe[target_col].values
    Xf_train, Xf_test, yfe_train, yfe_test = train_test_split(
        Xf.values, yfe, test_size=0.2, random_state=42, stratify=yfe
    )
    scaler = StandardScaler()
    Xf_train_s = scaler.fit_transform(Xf_train)
    Xf_test_s = scaler.transform(Xf_test)
    pca = PCA(n_components=0.95, random_state=42)
    Xf_train_p = pca.fit_transform(Xf_train_s)
    Xf_test_p = pca.transform(Xf_test_s)
    # Keras Tuner
    def build(hp):
        m = models.Sequential()
        for i in range(hp.Int('layers',1,2)):
            m.add(layers.Dense(hp.Int(f'units{i}',16,64,16), activation='relu'))
        if num_classes == 2:
            m.add(layers.Dense(1, activation='sigmoid'))
            m.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            m.add(layers.Dense(num_classes, activation='softmax'))
            m.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return m
    tuner = kt.RandomSearch(build, objective='val_accuracy', max_trials=3, directory=f'tmp_{target_col}', project_name='tmp')
    tuner.search(Xf_train_p, yfe_train, epochs=5, validation_split=0.1, verbose=0)
    best = tuner.get_best_models(num_models=1)[0]
    h_fe = best.fit(Xf_train_p, yfe_train, epochs=10, validation_split=0.1, verbose=0)
    yfe_pred = np.argmax(best.predict(Xf_test_p),axis=1) if num_classes>2 else (best.predict(Xf_test_p)>0.5).astype(int).flatten()
    cm_fe = confusion_matrix(yfe_test, yfe_pred)
    acc_fe = accuracy_score(yfe_test, yfe_pred)
    return df, cm_base, acc_base, cm_fe, acc_fe, h_base, h_fe, le_target.classes_

# Dataset icons
DATASET_ICONS = {
    'GlobalStore': '🛒',
    'Adult': '👤',
    'Diabetes': '🩺',
    'EA': '🏢',
    'Iris': '🌸',
}

# Dataset configurations
configs = {
    'GlobalStore': {
        'path':'Global Superstore.xlsx',
        'target':'Order Priority',
        'features':['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost', 'Segment', 'Ship Mode', 'Category', 'Sub-Category', 'Market', 'Region', 'Country', 'State', 'City'],
        'cats':['Segment', 'Ship Mode', 'Category', 'Sub-Category', 'Market', 'Region', 'Country', 'State', 'City'],
        'drop':['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Customer ID', 'Customer Name', 'Product ID', 'Product Name', 'Postal Code']
    },
    'Adult': {
        'path':'adult.csv',
        'target':'income',
        'features':['age', 'workclass', 'education.num', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country'],
        'cats':['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country'],
        'drop':['fnlwgt', 'education']
    },
    'Diabetes': {
        'path':'diabetes.csv',
        'target':'Outcome',
        'features':['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'],
        'cats':[],
        'drop':[]
    },
    'EA': {
        'path':'EA.csv',
        'target':'Attrition',
        'features':numeric_cols,
        'cats':categorical_cols,
        'drop':drop_cols
    },
    'Iris': {'path':'Iris.csv', 'target':'Species', 'features':feature_cols, 'cats':['Species'], 'drop':['Id']},
}

# Sidebar: dataset selection as radio buttons
st.sidebar.title("Choose Dataset")
dataset_names = list(configs.keys())
dataset_labels = [f"{DATASET_ICONS[name]} {name}" for name in dataset_names]
sel_idx = st.sidebar.radio("Dataset", options=range(len(dataset_names)), format_func=lambda i: dataset_labels[i])
sel = dataset_names[sel_idx]
conf = configs[sel]

# Load data based on file extension
file_path = conf['path']
if file_path.endswith('.xlsx'):
    df = pd.read_excel(file_path)
else:
    df = pd.read_csv(file_path)

df, cm_base, acc_base, cm_fe, acc_fe, h_base, h_fe, classes = load_and_train(
    df, conf['target'], conf['features'], conf['cats'], conf['drop']
)

# Main dashboard layout
st.markdown(f"# {DATASET_ICONS[sel]} <span style='color:#222'>{sel} Dataset Dashboard</span>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
col3, col4 = st.columns(2, gap="large")

with col1:
    st.markdown("## 🗂️ Data Preview")
    st.dataframe(df.head(), use_container_width=True)
with col2:
    st.markdown("## 📊 Baseline Confusion Matrix")
    st.plotly_chart(px.imshow(cm_base, text_auto=True, title="Baseline CM", color_continuous_scale='blues'), use_container_width=True)
with col3:
    st.markdown("## 🛠️ Feature Eng. Confusion Matrix")
    st.plotly_chart(px.imshow(cm_fe, text_auto=True, title="FE CM", color_continuous_scale='purples'), use_container_width=True)
with col4:
    st.markdown("## 🏆 Accuracy Comparison")
    st.bar_chart(pd.DataFrame({'Baseline':[acc_base],'FE':[acc_fe]}))

st.markdown("---")

# Curves section full width
st.markdown("## 📈 Training Curves")
curve_col1, curve_col2 = st.columns(2, gap="large")
with curve_col1:
    fig1 = px.line(y=h_base.history['accuracy'], title='Baseline Accuracy', labels={'y':'Accuracy', 'x':'Epoch'})
    fig1.add_scatter(y=h_base.history['val_accuracy'], name='Validation')
    fig1.update_layout(template='plotly_dark')
    st.plotly_chart(fig1, use_container_width=True)
with curve_col2:
    fig2 = px.line(y=h_fe.history['accuracy'], title='FE Accuracy', labels={'y':'Accuracy', 'x':'Epoch'})
    fig2.add_scatter(y=h_fe.history['val_accuracy'], name='Validation')
    fig2.update_layout(template='plotly_dark')
    st.plotly_chart(fig2, use_container_width=True)
