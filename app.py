
import streamlit as st
import joblib

# تحميل النموذج والـ vectorizer
model = joblib.load("svm_final_model.pkl")
vectorizer = joblib.load("svm_final_vectorizer.pkl")

st.set_page_config(page_title="كاشف جمع التكسير", page_icon="📚", layout="centered")
st.title("🧠 كاشف جمع التكسير الذكي")
st.markdown("تحقق هل الكلمة التي أدخلتها **جمع تكسير** أم لا باستخدام الذكاء الاصطناعي.")

word = st.text_input("🔤 أدخل الكلمة:", "")

if word:
    word_vec = vectorizer.transform([word])
    prediction = model.predict(word_vec)[0]
    prob = model.predict_proba(word_vec).max() * 100

    if prediction == "broken":
        st.success(f"✅ الكلمة '{word}' مصنفة كـ **جمع تكسير**. (الثقة: {prob:.2f}%)")
    else:
        st.info(f"ℹ️ الكلمة '{word}' **ليست جمع تكسير**. (الثقة: {prob:.2f}%)")

    st.markdown("---")
    st.caption("💡 النموذج يستخدم TF-IDF وتحليل نمط الحروف لكشف الجمع.")
