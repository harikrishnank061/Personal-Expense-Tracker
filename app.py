import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import cohere

# Streamlit app layout
st.set_page_config(page_title="Personal Expense Tracker", page_icon="💰")
st.title("💰 Personal Expense Tracker")
st.markdown("Track your daily expenses, visualize spending habits, and get AI-powered insights.")

# Input API key directly in the app
api_key = st.text_input("Enter your Cohere API Key", type="password")

# Initialize or load existing data
if 'expenses' not in st.session_state:
    st.session_state['expenses'] = []

# Input fields for expense entry
date = st.date_input("Date", datetime.date.today())
description = st.text_input("Description")
category = st.selectbox("Category", ["Food", "Transport", "Entertainment", "Utilities", "Other"])
amount = st.number_input("Amount ($)", min_value=0.01, step=0.01)

# Button to add expense
if st.button("Add Expense"):
    if description.strip():
        st.session_state['expenses'].append({
            'Date': date,
            'Description': description,
            'Category': category,
            'Amount': amount
        })
        st.success(f"✅ Expense '{description}' added successfully!")
    else:
        st.warning("⚠️ Please enter a valid description.")

# Display the expense table
if st.session_state['expenses']:
    df = pd.DataFrame(st.session_state['expenses'])
    st.subheader("Expense Summary")
    st.dataframe(df)
    
    # Show total expenses
    total_expense = df['Amount'].sum()
    st.write(f"**Total Expenses:** ${total_expense:.2f}")
    
    # Show expenses by category
    st.subheader("Expenses by Category")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df['Category'], y=df['Amount'], estimator=sum)
    plt.title("Total Expenses by Category")
    plt.xlabel("Category")
    plt.ylabel("Total Amount ($)")
    st.pyplot(plt)
    
    # Generate spending insights using Cohere
    if api_key:
        try:
            co = cohere.Client(api_key)
            expense_summary = df.groupby('Category')['Amount'].sum().reset_index()
            summary_text = ", ".join(
                [f"{row['Category']}: ${row['Amount']:.2f}" for _, row in expense_summary.iterrows()]
            )
            prompt = f"Generate a spending summary based on the following categories and amounts: {summary_text}."
            response = co.generate(
                model="command-xlarge",
                prompt=prompt,
                max_tokens=300,
                temperature=0.7,
                k=0,
                p=0.8,
                frequency_penalty=0.2,
                presence_penalty=0.2
            )
            insights = response.generations[0].text.strip()
            st.subheader("Spending Insights:")
            st.write(insights)
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
else:
    st.info("No expenses added yet. Start tracking your spending!")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Cohere AI")
