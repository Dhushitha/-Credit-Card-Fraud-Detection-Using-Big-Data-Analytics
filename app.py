import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection Using Big Data Analytics")

uploaded_file = st.file_uploader("Upload Credit Card Fraud Dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ---------------- DATASET OVERVIEW ----------------
    st.header("1️⃣ Dataset Overview")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.dataframe(df.head())

    # ---------------- DATA TYPES ----------------
    st.header("2️⃣ Column Data Types")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))

    # ---------------- DATA QUALITY ----------------
    st.header("3️⃣ Data Quality Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum())

    with col2:
        st.subheader("Duplicate Rows")
        st.write("Duplicates:", df.duplicated().sum())

    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.drop_duplicates(inplace=True)

    # ---------------- CLASS IMBALANCE ----------------
    st.header("4️⃣ Fraud vs Legitimate Distribution")
    class_counts = df["Class"].value_counts().sort_index()
    st.dataframe(class_counts)

    fig1 = plt.figure()
    class_counts.plot(kind="bar")
    plt.xlabel("Class (0 = Legit, 1 = Fraud)")
    plt.ylabel("Count")
    plt.title("Fraud vs Legitimate Transactions")
    st.pyplot(fig1)

    # ---------------- DESCRIPTIVE STATS ----------------
    st.header("5️⃣ Descriptive Statistics")
    st.dataframe(df.describe())

    # ---------------- AMOUNT ANALYSIS ----------------
    st.header("6️⃣ Transaction Amount Analysis")

    col3, col4 = st.columns(2)

    with col3:
        avg_amt = df.groupby("Class")["Amount"].mean()
        fig2 = plt.figure()
        avg_amt.plot(kind="bar")
        plt.title("Average Transaction Amount by Class")
        st.pyplot(fig2)

    with col4:
        fig3 = plt.figure()
        df["Amount"].hist(bins=50)
        plt.title("Overall Amount Distribution")
        st.pyplot(fig3)

    # ---------------- FRAUD AMOUNT ----------------
    st.header("7️⃣ Fraud Amount Distribution")

    fig4 = plt.figure()
    df[df["Class"] == 1]["Amount"].hist(bins=30)
    plt.title("Fraud Transaction Amount Distribution")
    st.pyplot(fig4)

    # ---------------- HIGH VALUE FRAUD ----------------
    st.header("8️⃣ High Value Fraud Transactions")
    st.dataframe(df[df["Class"] == 1].sort_values("Amount", ascending=False).head(10))

    # ---------------- CORRELATION ----------------
    st.header("9️⃣ Correlation Analysis")
    numeric_cols = df.select_dtypes(include="number").columns[:10]
    corr = df[numeric_cols].corr()

    fig5 = plt.figure(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    st.pyplot(fig5)

    # ---------------- METRICS ----------------
    st.header("🔟 Key Metrics")
    fraud_percentage = (class_counts.get(1,0) / len(df)) * 100
    st.metric("Fraud Percentage", f"{fraud_percentage:.4f}%")
    st.metric("Total Transactions", len(df))
    st.metric("Fraud Transactions", class_counts.get(1,0))

    # ---------------- INSIGHTS ----------------
    st.header("💡 Insights & Observations")
    st.markdown("""
    - Fraud transactions are extremely rare compared to legitimate ones  
    - Fraudulent transactions tend to have irregular amount patterns  
    - Dataset is highly imbalanced, requiring specialized analytics  
    - High-value fraud transactions pose significant financial risk  
    - Correlation patterns can help identify suspicious behavior
    """)

    st.success("✅ Complete Credit Card Fraud Analysis Generated Successfully")