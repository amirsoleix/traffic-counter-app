import os

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import datetime
import plotly.express as px
import plotly.graph_objects as go
from persiantools.jdatetime import JalaliDate

st.set_page_config(page_title="Traffic Counter Data", page_icon="😎")
# st.set_config('browser.uiDirection', 'RTL')

with open("./style.css") as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

@st.cache_resource
def load_data():
    path = "data/data.csv.zip"
    if not os.path.isfile(path):
        # path = f"Path to GitHub repository"
        path = None

    data = pd.read_csv(
        path,
        names = [
            'road code', 'road name', 'start time', 'end time', 'operation length (minutes)', 'class 1', 'class 2',
            'class 3', 'class 4', 'class 5', 'estimated number', 'province', 'start city', 'end city', 'edge name'
        ],
        skiprows=1,
        usecols=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19]
    )
    # Change "start time" to date using Jalali
    data['start date'] = data['start time'].apply(lambda x: JalaliDate(
        int(x.split(' ')[0].split('/')[0]),
        int(x.split(' ')[0].split('/')[1]),
        int(x.split(' ')[0].split('/')[2])
    ))

    data['end date'] = data['end time'].apply(lambda x: JalaliDate(
        int(x.split(' ')[0].split('/')[0]),
        int(x.split(' ')[0].split('/')[1]),
        int(x.split(' ')[0].split('/')[2])
    ))

    return data

@st.cache_resource
def load_coordinates():
    path = "data/coordinates.csv.zip"
    if not os.path.isfile(path):
        # path = f"Path to GitHub repository"
        path = None
    
    coordinates = pd.read_csv(
        path,
        names=[
            "city",
            "lat",
            "lon"
        ],
        skiprows=1,
        usecols=[0, 1, 2]
    )

    return coordinates



st.title("شمال: قصه‌ای برای نوروز")
st.write("""
‫شمال ایران با طبیعت بکر آب و هوای دل‌پذیرش همواره یکی از مقاصد گردشگری محبوب بوده است. از رشت تا مازندران، از تالش تا آستارا، هر ناحیه‌ای در این منطقه دارای جذابیت‌های منحصر به فردی است که هر ساله گردشگران بسیاری را جذب می‌کند.
""")
st.write("""
در این پروژه، ما قصد داریم با بررسی داده های تردد شمار به  شناسایی و معرفی پرطرفدارترین شهرهای شمال ایران بپردازیم. این تحقیق به هدف ارائه اطلاعاتی کامل و مفید به گردشگران و علاقه‌مندان به سفر در این منطقه، و همچنین به عنوان یک منبع برای محققین و برنامه‌ریزان گردشگری و توسعه گردشگری ارائه می‌شود.
""")
st.header("""آغاز سفر""")
st.write("""
پس از اتمام محدودیت‌ها و موازنه‌های طولانی‌ای که برای کنترل شیوع کرونا صورت گرفت، به نظر می‌رسد که جامعه به سمت بازگشت به زندگی عادی‌تر و فعالیت‌های روزمره خود پیش رفته است. داده‌ها نشان می دهند که با وجود محدودیت های اندک در سال هزار و چهارصد سفرها به روال عادی خود برگشته و پس از رفع آنها در سال بعدی حتی مسافران بیشتری به این منطقه جذب شده‌اند.
""")