import streamlit as st
import pandas as pd
import joblib

def main():
    st.set_page_config(page_title="Professor Salary Predictor", layout="centered")

    st.title("🎓 Professor Salary Predictor")
    st.write("Enter the details below to predict the professor's salary.")

    # Load the newly trained model
    try:
        model = joblib.load('trained_model.joblib')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please run the 'train_model.py' script first to generate a compatible model file.")
        return

    col1, col2 = st.columns(2)

    with col1:
        rank = st.selectbox("Rank", options=['Prof', 'AssocProf', 'AsstProf'])
        phd = st.number_input("Years since PhD", min_value=0, max_value=70, value=20)
        sex = st.selectbox("Sex", options=['Male', 'Female'])

    with col2:
        discipline = st.selectbox("Discipline", options=['A', 'B'])
        service = st.number_input("Years of Service", min_value=0, max_value=70, value=15)

    if st.button("Predict Salary"):
        # Create input dataframe with correct column names
        input_data = pd.DataFrame({
            'rank': [rank],
            'discipline': [discipline],
            'phd': [phd],
            'service': [service],
            'sex': [sex]
        })

        try:
            prediction = model.predict(input_data)
            salary = prediction[0]
            st.success(f"### Predicted Salary: ${salary:,.2f}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == '__main__':
    main()
