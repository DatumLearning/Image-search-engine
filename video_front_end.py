import streamlit as st
import numpy as np
from PIL import Image
import time
from scipy.spatial.distance import cdist

@st.cache_data
def read_data():
    all_vecs = np.load("all_vecs.npy")
    all_names = np.load("all_names.npy")
    return all_vecs , all_names

vecs , names = read_data()

_ , fcol2 , _ = st.columns(3)

scol1 , scol2 = st.columns(2)

ch = scol1.button("Start / change")
fs = scol2.button("find similar")

if ch:
    random_name = names[np.random.randint(len(names))]
    fcol2.image(Image.open("./images/" + random_name))
    st.session_state["disp_img"] = random_name
    st.write(st.session_state["disp_img"])
if fs:
    c1 , c2 , c3 , c4 , c5 = st.columns(5)
    idx = int(np.argwhere(names == st.session_state["disp_img"]))
    target_vec = vecs[idx]
    fcol2.image(Image.open("./images/" + st.session_state["disp_img"]))
    top5 = cdist(target_vec[None , ...] , vecs).squeeze().argsort()[1:6]
    c1.image(Image.open("./images/" + names[top5[0]]))
    c2.image(Image.open("./images/" + names[top5[1]]))
    c3.image(Image.open("./images/" + names[top5[2]]))
    c4.image(Image.open("./images/" + names[top5[3]]))
    c5.image(Image.open("./images/" + names[top5[4]]))










    
