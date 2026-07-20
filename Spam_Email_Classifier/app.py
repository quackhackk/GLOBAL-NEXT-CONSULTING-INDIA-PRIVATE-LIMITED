#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import joblib

model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Spam Email Classifier")

email = st.text_area("Enter Email")

if st.button("Predict"):
    x = vectorizer.transform([email])
    pred = model.predict(x)[0]
    prob = model.predict_proba(x)[0]

    if pred == 1:
        st.error(f"Spam ({prob[1]*100:.2f}%)")
    else:
        st.success(f"Ham ({prob[0]*100:.2f}%)")


# In[2]:





# In[5]:


import sys
print(sys.executable)


# In[ ]:




