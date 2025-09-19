import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app layout
st.set_page_config(page_title="Email Spam Detector", page_icon="📧")
st.title("📧 Email Spam Detector")
st.write("Enter an email message below to check if it is **ham** or **spam**.")

# Email input
email_input = st.text_area("Enter your email message here:")

if st.button("Predict"):
    if not email_input.strip():
        st.warning("Please enter an email message!")
    else:
        # Transform input using saved TF-IDF vectorizer
        email_tfidf = vectorizer.transform([email_input])
        
        # Predict probability and class
        proba = model.predict_proba(email_tfidf)
        prediction = model.predict(email_tfidf)[0]
        
        # Display results
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("⚠️ Predicted Class: SPAM")
        else:
            st.success("✅ Predicted Class: HAM")
        
        st.write(f"Probability of HAM: {proba[0][0]:.4f}")
        st.write(f"Probability of SPAM: {proba[0][1]:.4f}")

st.markdown("---")
st.write("Model trained using Logistic Regression with SMOTE handling imbalanced dataset.")
