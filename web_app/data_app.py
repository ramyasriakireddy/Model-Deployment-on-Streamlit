import streamlit as st

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
from PIL import Image


dataset_loc = "data/train.csv"
image_loc = "img/SVM-kernels.jpg"
img_svm = Image.open(image_loc).convert('RGB')


@st.cache
def load_data(dataset_loc):
    df = pd.read_csv(dataset_loc)
    return df


def load_description(df):
        # Preview of the dataset
        st.header("Data Preview")
        preview = st.radio("Choose Head/Tail?", ("Top", "Bottom"))
        if(preview == "Top"):
            st.write(df.head())
        if(preview == "Bottom"):
            st.write(df.tail())

        # display the whole dataset
        if(st.checkbox("Show complete Dataset")):
            st.write(df)

        # Show shape
        if(st.checkbox("Display the shape")):
            st.write(df.shape)
            dim = st.radio("Rows/Columns?", ("Rows", "Columns"))
            if(dim == "Rows"):
                st.write("Number of Rows", df.shape[0])
            if(dim == "Columns"):
                st.write("Number of Columns", df.shape[1])

        # show columns
        if(st.checkbox("Show the Columns")):
            st.write(df.columns)

def load_viz(df):
    fig = plt.figure()
    sns.countplot(x='output', data=df)
    fig.set_figheight(3)
    fig.set_figwidth(5)
    st.pyplot(fig)

    st.title("Pair Plot")
    fig = sns.pairplot(df, hue="output")
    fig.fig.set_figheight(6)
    fig.fig.set_figwidth(10)
    st.pyplot(fig)


def main():

    # Title/ text
    st.title('Classification')
    st.image(img_svm, use_column_width = True)


    # loading the data
    df = load_data(dataset_loc)

    # display description
    load_description(df)

    load_viz(df)



if(__name__ == '__main__'):
    main()
