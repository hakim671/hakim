import streamlit as st
import pandas as pd

st.title('🎈 Hakim app')

st.write('Hakim vret!')

with st.expander("initialdata"):
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  st.write("**x**")
  X_raw = df.drop("species",axis=1)
  X_raw
  st.write("**y**")
  y_raw = df.species
  y_raw
