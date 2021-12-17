import functools
import time
import PIL.Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import IPython.display as display
import streamlit as st
import pandas as pd
import numpy as np
from selenium import webdriver

from style_transfer_model import StyleTransferModel, get_images, tensor_to_image
from image_scrapper import search_and_download

import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


# Sidebar
selectbox1 = st.sidebar.selectbox(
    "What model do you want to use",
    ("VGG19", "VGG16", "Resnet"))

add_selectbox2 = st.sidebar.selectbox(
    "What optimizer do you want to use",
    ("SGD", "SD", "ADAM"))


# Title
st.title('Style transfer')
st.subheader('No need to be an artist, Just copy their style')
st.subheader('')


# Import image (######### Modify the path #########)
st.header("1. Select the image")
content_path = st.file_uploader("Upload the picture you want to add style to")
if content_path is not None:
    st.image(content_path)
    content_path = "/Users/edouardlabarthe/Downloads/" + str(content_path.name)


# Import style
st.subheader('')
st.header("2. Select the style")
col1, col2 = st.columns(2)

# 1.Import picture (######### Modify the path #########)
with col1:
    st.subheader("2.1 From an image")
    style_path = st.file_uploader(
        "Upload the picture you want to copy the style")
    if style_path is not None:
        st.write(style_path)
        st.image(style_path)
        style_path = "/Users/edouardlabarthe/Downloads/" + str(style_path.name)


# 2.Import name of peinter
with col2:
    st.subheader("2.2 From a painter's name")
    painter = st.text_input('Name a painters')
    nb_paintings = st.slider("Number of painting for artist to use", 1, 5, 1)

    clicked2 = st.button("Find arts")

    ####### SELECT OWN PATH FOR CHROMEDRIVER #########
    if clicked2 == True:
        keyword = painter + " paintings"
        DRIVER_PATH = '/Users/edouardlabarthe/Downloads/chromedriver'
        wd = webdriver.Chrome(executable_path=DRIVER_PATH)
        style_path = search_and_download(
            wd=wd, search_term=keyword, number_images=1)
        st.write(style_path)
    # st.image(style_path)

    if style_path is not None:
        st.image(style_path)

    #choice = st.multiselect("Select good pictures", [1, 2, 3, 4])


# Output
st.header("3. Result")
st.subheader("")

n_epoch = st.slider("Select the number of epochs", 1, 15, 1)
n_step = st.slider("Select the level of steps per epochs", 1, 15, 1)

clicked = st.button("RUN")

if clicked == True:
    content_image, style_image = get_images(content_path, style_path)

    model = StyleTransferModel()
    stylized_image = model(content_image, style_image,
                           n_epochs=n_epoch, n_steps_per_epoch=n_step)

    stylized_image = tensor_to_image(stylized_image)
    st.image(stylized_image)
    st.download_button('Dowload the picture', stylized_image)

number = st.slider("Select the level of style")
