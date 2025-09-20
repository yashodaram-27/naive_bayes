import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.title("Salary Prediction App")
st.write("This app predicts whether a person's salary is <=50K or >50K based on their attributes.")

# A function to load data and train the model
# This is wrapped in a function with caching to prevent re-training on every user interaction
@st.cache_data
def load_data_and_train_model():
    # Load the datasets
    try:
        train_data = pd.read_csv(r'C:\Users\Win\OneDrive\Documents\naive_bayes\SalaryData_Train.csv')
        test_data = pd.read_csv(r'C:\Users\Win\OneDrive\Documents\naive_bayes\SalaryData_Train.csv')
    except FileNotFoundError:
        st.error("Dataset files not found. Please ensure 'SalaryData_Train.csv' and 'SalaryData_Test.csv' are in the same directory.")
        st.stop()

    # Concatenate for consistent preprocessing
    df = pd.concat([train_data, test_data], ignore_index=True)
    df.drop_duplicates(inplace=True)

    # Pre-processing
    df['Salary'] = df['Salary'].map({'<=50K': 0, '>50K': 1})
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()

    numerical_cols = ['age', 'educationno', 'capitalgain', 'capitalloss', 'hoursperweek']
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # Scale numerical features
    scaler = StandardScaler()
    scaler.fit(df[numerical_cols])

    # Encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Split the data back
    X_train = df.iloc[:train_data.shape[0]].drop('Salary', axis=1)
    y_train = df.iloc[:train_data.shape[0]]['Salary']

    # Apply scaling to training data
    X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])

    # Train the model
    gnb_model = GaussianNB()
    gnb_model.fit(X_train, y_train)

    return gnb_model, scaler, label_encoders, numerical_cols, categorical_cols, df

# Load model, scaler, and encoders
model, scaler, label_encoders, numerical_cols, categorical_cols, full_df = load_data_and_train_model()

# User input section
st.sidebar.header("Enter Your Details")
age = st.sidebar.slider("Age", 17, 90, 30)
workclass = st.sidebar.selectbox("Workclass", options=full_df['workclass'].unique())
education = st.sidebar.selectbox("Education", options=full_df['education'].unique())
educationno = st.sidebar.slider("Education Years", 1, 16, 13)
maritalstatus = st.sidebar.selectbox("Marital Status", options=full_df['maritalstatus'].unique())
occupation = st.sidebar.selectbox("Occupation", options=full_df['occupation'].unique())
relationship = st.sidebar.selectbox("Relationship", options=full_df['relationship'].unique())
race = st.sidebar.selectbox("Race", options=full_df['race'].unique())
sex = st.sidebar.selectbox("Sex", options=full_df['sex'].unique())
capitalgain = st.sidebar.number_input("Capital Gain", min_value=0, value=0, step=100)
capitalloss = st.sidebar.number_input("Capital Loss", min_value=0, value=0, step=10)
hoursperweek = st.sidebar.slider("Hours per Week", 1, 99, 40)
native = st.sidebar.selectbox("Native Country", options=full_df['native'].unique())

# Create a dictionary for the new data
new_data = {
    'age': [age],
    'workclass': [workclass],
    'education': [education],
    'educationno': [educationno],
    'maritalstatus': [maritalstatus],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'sex': [sex],
    'capitalgain': [capitalgain],
    'capitalloss': [capitalloss],
    'hoursperweek': [hoursperweek],
    'native': [native]
}
new_df = pd.DataFrame(new_data)

# Make prediction when button is clicked
if st.sidebar.button("Predict Salary"):
    # Preprocess the new data
    for col in categorical_cols:
        new_df[col] = label_encoders[col].transform(new_df[col])
    
    new_df[numerical_cols] = scaler.transform(new_df[numerical_cols])

    # Reorder columns to match the training data
    new_df = new_df[numerical_cols + categorical_cols]

    prediction = model.predict(new_df)
    
    if prediction[0] == 1:
        st.success("The predicted salary is: **>50K**")
    else:
        st.success("The predicted salary is: **<=50K**")

st.markdown("---")
st.markdown("### How to Run this App")
st.markdown("1. Save the code as `app.py`.")
st.markdown("2. Make sure your datasets (`SalaryData_Train.csv`, `SalaryData_Test.csv`) are in the same directory.")
st.markdown("3. Open your terminal and navigate to the directory.")
st.markdown("4. Run the command: `streamlit run app.py`")
st.markdown("5. The app will open in your browser.")
