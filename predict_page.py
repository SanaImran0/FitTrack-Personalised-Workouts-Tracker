import streamlit as st 
import pickle 
import numpy as np  


def load_model():
    with open('', 'rb') as file:
        data = pickle.load(file)       
    return data    

data = load_model()

def show_streamlit_