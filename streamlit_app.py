import streamlit as st
import pandas as pd

st.title('🎈 Hakim app')

st.write('Zoir_daun!')

with st.expander("initial_data"):
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  st.write("**x**")
  X_raw = df.drop("species",axis=1)
  X_raw
  st.write("**y**")
  y_raw = df.species
  y_raw
with st.expander("data-viz"):
  st.scatter_chart(data=df,x="bill_depth_mm",y='body_mass_g', color = 'species')
  st.scatter_chart( data= df, x = 'bill_depth_mm', y = 'sex', color = 'species')
  st.scatter_chart(data=df, x='bill_depth_mm', y='sex', color='species')
