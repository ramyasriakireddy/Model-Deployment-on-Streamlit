import streamlit as st
import data_app as da
import ml_app as ma

def main():
    
    da.main()
    st.header("RBF SVM Predictor")
    ma.main()

if(__name__ == '__main__'):
    main()
