import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

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

with open("./style.css") as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

@st.cache_resource
def load_data():
    path = "data/data.csv.zip"
    if not os.path.isfile(path):
        path = "https://github.com/amirsoleix/traffic-counter-app/blob/master/data/data.csv.zip"

    data = pd.read_csv(
        path,
        names = [
            'road code', 'road name', 'start time', 'end time', 'operation length (minutes)', 'class 1', 'class 2',
            'class 3', 'class 4', 'class 5', 'estimated number', 'province', 'start city', 'end city', 'edge name'
        ],
        skiprows=1,
        usecols=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19]
    )
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
        path = "https://github.com/amirsoleix/traffic-counter-app/blob/master/data/coordinates.csv.zip"

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

@st.cache_resource
def load_test():
    path = "data/hypothesis-test.csv"
    if not os.path.isfile(path):
        path = "https://github.com/amirsoleix/traffic-counter-app/blob/master/data/hypothesis-test.csv"

    coordinates = pd.read_csv(
        path,
        names=[
            "city",
            "entries"
        ],
        skiprows=1,
        usecols=[0, 1]
    )

    return coordinates

@st.cache_resource
def load_population():
    path = "data/population.csv.zip"
    if not os.path.isfile(path):
        path = "https://github.com/amirsoleix/traffic-counter-app/blob/master/data/population.csv.zip"
    
    population = pd.read_csv(
        path,
        names=[
            "city",
            "population"
        ],
        skiprows=1,
        usecols=[0, 1]
    )

    return population

@st.cache_resource
def cities_chart(data, cities, graph_title):
    years = [
        [JalaliDate(1395, 12, 20), JalaliDate(1396, 1, 18)],
        [JalaliDate(1396, 12, 20), JalaliDate(1397, 1, 18)],
        [JalaliDate(1397, 12, 20), JalaliDate(1398, 1, 18)],
        [JalaliDate(1398, 12, 20), JalaliDate(1399, 1, 18)],
        [JalaliDate(1399, 12, 20), JalaliDate(1400, 1, 18)],
        [JalaliDate(1400, 12, 20), JalaliDate(1401, 1, 18)],
        [JalaliDate(1401, 12, 20), JalaliDate(1402, 1, 18)],
    ]

    # Initialize the DataFrame with dynamic city names
    df = pd.DataFrame(columns=["Year"] + cities)

    for year in years:
        count = []
        for city in cities:
            total_count = int(data[(data['start date'] >= year[0]) & (data['end date'] <= year[1]) & (data['end city'] == city)]['class 1'].sum()) + int(data[(data['start date'] >= year[0]) & (data['end date'] <= year[1]) & (data['end city'] == city)]['class 2'].sum())
            count.append(total_count)
        year_data = [year[0].year + 1] + count
        df = pd.concat([df, pd.DataFrame([year_data], columns=["Year"] + cities)])

    # Plotting the data for each city
    fig = go.Figure()
    colors = ['indianred', 'lightsalmon', 'lightseagreen']  # You can extend this list for more cities.
    for idx, city in enumerate(cities):
        fig.add_trace(go.Bar(
            x=df['Year'],
            y=df[city],
            name=city,
            marker_color=colors[idx % len(colors)]
        ))

    fig.update_layout(
        barmode='group',
        xaxis_tickangle=0,
        title=graph_title,
        xaxis_title="Year",
        yaxis_title="Incoming Vehicle Count",
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

@st.cache_data
def find_dates():
    dates = []
    for year in range(1399, 1403):
        for month in [1, 12]:
            if year == 1402 and month == 12:
                continue
            if month == 12:
                for day in range(20, 31):
                    try:
                        JalaliDate(year, month, day)
                        dates.append(f"{year}-{month}-{day}")
                    except:
                        pass
            if month == 1:
                for day in range(1, 20):
                    try:
                        JalaliDate(year, month, day)
                        dates.append(f"{year}-{month}-{day}")
                    except:
                        pass
    return dates

@st.cache_data
def city_input(data, start_date, end_date):
    start_date = JalaliDate(
        int(start_date.split("-")[0]), int(start_date.split("-")[1]), int(start_date.split("-")[2]
    ))
    end_date = JalaliDate(
        int(end_date.split("-")[0]), int(end_date.split("-")[1]), int(end_date.split("-")[2]
    ))
    data = data[(data["start date"] >= start_date) & (data["end date"] <= end_date)]
    city_input = pd.DataFrame(columns=["date", "city", "input", "output", "net", "tourist", "province"])
    cities = data["start city"].unique()
    date_range = pd.date_range(start_date.to_gregorian(), end_date.to_gregorian())
    jalali_range = [JalaliDate(date) for date in date_range]

    for city in cities:
        for date in jalali_range:
            city_input = pd.concat([city_input, pd.DataFrame({
                "date": [date],
                "city": [city],
                "input": [0],
                "output": [0],
                "net": [0],
                "tourist": [0],
                "province": [data[data["start city"] == city]["province"].values[0]]
            })])

    for _, row in data.iterrows():
        city_input.loc[ (city_input["city"] == row["start city"]) & (city_input["date"] == row["start date"]), "output"] += row["class 1"]
        city_input.loc[ (city_input["city"] == row["end city"]) & (city_input["date"] == row["end date"]), "input"] += row["class 1"]

    city_input["net"] = city_input["input"] - city_input["output"]
    city_input["tourist"] = city_input["net"].apply(lambda x: abs(x))
    city_input = city_input.groupby("city").agg({
        "input": "sum",
        "output": "sum",
        "net": "sum",
        "tourist": "sum",
        "province": "first"
    }).reset_index()
    city_input = city_input.merge(population, on="city", how="left")
    city_input.sort_values(by=['tourist'], inplace=True, ascending=False)
    city_input.reset_index(inplace=True)

    return city_input

def map(data, lat, lon, zoom):
    st.write(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state={
                "latitude": lat,
                "longitude": lon,
                "zoom": zoom,
                "pitch": 50,
            },
            layers=[
                pdk.Layer(
                    "HexagonLayer",
                    data=data,
                    get_position=["lon", "lat"],
                    get_elevation_weight="tourist",
                    get_color_weight="tourist",
                    color_range=[
                        [215,48,39],
                        [252,141,89],
                        [254,224,139],
                        [217,239,139],
                        [145,207,96],
                        [26,152,80]
                    ],
                    radius=5000,
                    elevation_scale=6,
                    opacity=0.8,
                    elevation_range=[0, 8000],
                    pickable=True,
                    extruded=True,
                ),
            ],
        )
    )

@st.cache_data
def aggregate_data(aggregate_data):
    aggregate_data["input per capita"] = aggregate_data["input"] / (aggregate_data["population"] + 1)
    aggregate_data["output per capita"] = aggregate_data["output"] / (aggregate_data["population"] + 1)
    aggregate_data["net per capita"] = aggregate_data["net"] / (aggregate_data["population"] + 1)
    aggregate_data["tourist per capita"] = aggregate_data["tourist"] / (aggregate_data["population"] + 1)
    return aggregate_data

@st.cache_data
def dbscan(data, year):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    data = data[~data["city"].isin([
        'Ghazvin',
        'Gilan Border',
        'Golestan Forest',
        'Imamzadeh Hashem Tehran',
        'Kandovan',
        'Park Jangali',
        'Se-rah-e Kalaleh',
        'Sisangan'
    ])]
    X = data[["tourist per capita", "tourist"]].values
    X = StandardScaler().fit_transform(X)

    # Cluster to 3 clusters
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    data["cluster"] = db.labels_

    fig = px.scatter(data, x="tourist", y="tourist per capita", color="cluster", hover_name="city", template='plotly', height=300)
    fig.update_layout(title=f"DBSCAN Clustering of Cities for {year}")
    fig.update_layout(title_font_size=14)
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(
        yaxis_type="log"
    )
    st.plotly_chart(fig)

    return data

@st.cache_data
def kmeans(data, year):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    data = data[~data["city"].isin([
        'Ghazvin',
        'Gilan Border',
        'Golestan Forest',
        'Imamzadeh Hashem Tehran',
        'Kandovan',
        'Park Jangali',
        'Se-rah-e Kalaleh',
        'Sisangan'
    ])]
    X = data[["tourist per capita", "tourist"]].values
    X = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=8).fit(X)
    data["cluster"] = kmeans.labels_

    fig = px.scatter(data, x="tourist", y="tourist per capita", color="cluster", hover_name="city", template='plotly', height=300)
    fig.update_layout(title=f"K-Means Clustering of Cities for {year}")
    fig.update_layout(title_font_size=14)
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(
        yaxis_type="log"
    )
    st.plotly_chart(fig)

    return data

@st.cache_data
def gmm(data, year):
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    data = data[~data["city"].isin([
        'Ghazvin',
        'Gilan Border',
        'Golestan Forest',
        'Imamzadeh Hashem Tehran',
        'Kandovan',
        'Park Jangali',
        'Se-rah-e Kalaleh',
        'Sisangan'
    ])]
    X = data[["tourist per capita", "tourist"]].values
    X = StandardScaler().fit_transform(X)

    gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
    data["cluster"] = gmm.predict(X)

    # fig = px.scatter(data, x="tourist", y="tourist per capita", color="cluster", hover_name="city", template='plotly') Make it smaller
    fig = px.scatter(data, x="tourist", y="tourist per capita", color="cluster", hover_name="city", template='plotly', height=300)
    fig.update_layout(title=f"GMM Clustering of Cities for {year}")
    fig.update_layout(title_font_size=14)
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(
        yaxis_type="log"
    )
    st.plotly_chart(fig)

    return data

def update_query_params():
    date = st.session_state["date"]
    start_date = date[0]
    end_date = date[1]
    st.experimental_set_query_params(start_date=start_date, end_date=end_date)

data = load_data()
coordinates = load_coordinates()
population = load_population()

st.title("شمال: قصه‌ای برای نوروز")
st.write("""
‫شمال ایران با طبیعت بکر آب و هوای دل‌پذیرش همواره یکی از مقاصد گردشگری محبوب بوده است. از رشت تا مازندران، از تالش تا آستارا، هر ناحیه‌ای در این منطقه دارای جذابیت‌های منحصر به فردی است که هر ساله گردشگران بسیاری را جذب می‌کند.
""")
st.image("./images/header-image.jpg", use_column_width=True)
st.write("""
در این پروژه، ما قصد داریم با بررسی داده های تردد شمار به  شناسایی و معرفی پرطرفدارترین شهرهای شمال ایران بپردازیم. این تحقیق به هدف ارائه اطلاعاتی کامل و مفید به گردشگران و علاقه‌مندان به سفر در این منطقه، و همچنین به عنوان یک منبع برای محققین و برنامه‌ریزان گردشگری و توسعه گردشگری ارائه می‌شود.
""")
st.header("""\"بزن بریم شمال\"""")
st.write("""‫همه ساله استان های شمالی کشور انتخاب بسیاری از ایرانیان در تعطیلات است اما بدون شک، بهار شمال چیز دیگری است! این فصل به همراه آغاز رونق طبیعت، آب و هوای معتدل و روزهای آفتابی، جذابیت خاصی به سفرها در این مناطق می‌بخشد. تعطیلات نوروز نیز بهانه‌ای عالی برای فرار از روزمرگی و پرداختن به سفرهای خانوادگی و دوستانه است 
""")

cities_chart(data, ['Rasht', 'Sari', 'Gorgan'], 'Incoming Cars and Buses for Province Centers')
st.write("""
با تحلیل تعداد ترددها که داده‌های تردد شمار در اختیار ما قرار داده است ، مشخص شده که شمال کشور حتی در دوران کرونا محبوبیت خود را به عنوان یک مقصد برای رفع خستگی سالانه حفظ کرده است.
همچنین پس از گذشت دوران طولانی اعمال محدودیت‌ها برای کنترل شیوع کرونا ، به نظر می رسد که جامعه به سمت زندگی پیش از کرونا برگشته است.
""")

st.write("""
شاید فکر کنید که شهر های با جمعیت تقریبا یکسان تجربه مشابهی از مسافرت نوروزی  ارائه می‌دهند.اگرباور شما چنین است به نمودار زیر توجه کنید:
""")

cities_chart(data, ['Astaneh', 'Asalem', 'Kiyakala'], 'Incoming Cars and Buses for Cities with Similar Population')

st.write("""
شهر‌های آستانه ، کیاکلا و اسالم به ترتیب با جمعیت 7166 و 8140 و 10720 توزیع گردشگری  یکسانی نداشته و مثال نقضی بر این باورهستند
و نشان می دهند مسافران مقصد خود را با معیار
های دیگر و به صورت هوشمندانه تری انتخاب می کنند.

در ادامه، ما به دنبال تحلیل و معرفی شهرهای از شمال کشور هستیم که در سه سال گذشته از محبوبیت بیشتری در میان مسافران  نوروزی برخوردار بوده‌اند، و از روش‌های مختلفی برای این تحلیل استفاده می‌کنیم. سپس، به بررسی مناطقی می‌پردازیم که پتانسیل گردشگری مناسبی دارند و شاهد توسعه و رشد این صنعت در این نواحی هستیم. 
‫
""")

st.header("روش‌های بررسی")
st.write("""
استفاده از روش‌های آماری مانند خوشه‌بندی (clustering) و تست فرضیه (hypothesis testing) می‌تواند درک عمیق‌تری از داده‌ها و ترجیحات مسافران به ما بدهد. در ادامه، روش‌های مورد استفاده را به طور خلاصه توضیح می‌دهیم:
""")
st.subheader("روش خوشه‌بندی (Clustering)")
st.write("""
خوشه‌بندی یکی از تکنیک‌های یادگیری ماشین است که هدف آن دسته‌بندی داده‌ها بر اساس شباهت‌های میان آن‌ها است. در مورد تحلیل شهرهای شمال ایران، ما می‌توانیم از خوشه‌بندی برای گروه‌بندی شهرها بر اساس معیارهایی مانند تعداد گردشگران، نسبت گردشگر به جمعیت و نرخ ورود به شهرها استفاده کنیم. این تکنیک‌ها به ما کمک می‌کند تا شهرهایی که الگوهای مشابهی در جذب گردشگر دارند را شناسایی کنیم.
""")

st.subheader("تست فرض (Hypothesis Testing)")
st.write("""
تست فرض یک روش آماری است که برای ارزیابی صحت یک فرضیه بر اساس داده‌های موجود استفاده می‌شود. برای مثال، اگر می‌خواهیم بدانیم آیا تفاوت معنی‌داری بین تعداد گردشگران در دو شهر خاص در شمال ایران وجود دارد یا خیر، می‌توانیم از تست فرض استفاده کنیم. این روش به ما امکان می‌دهد تا با اطمینان بیشتری تصمیم‌گیری کنیم و مشخص کنیم که آیا تفاوت‌های مشاهده‌شده تصادفی هستند یا ناشی از عوامل بنیادین در جذب گردشگران.
""")

st.subheader("نسبت توریست‌های شهرهای مختلف")
st.write("""
در اینجا محیطی برای آشنایی بیشتر شما تعبیه شما که می‌توانید با مشخص‌کردن تاریخ مدنظر خود، تعداد سواری و مینی‌بوس‌های ورودی، خروجی و تعداد اتومبیل‌هایی که احتمال می‌رود مربوط به گردشگران باشد (چرا که حداقل یک روز در شهر توقف داشته‌اند) را مشاهده کنید.
""")
start_date, end_date = st.select_slider(
    'Select the date range for analysis.',
    options=find_dates(),
    key='date',
    value=("1400-12-25", "1401-1-18"),
    on_change=update_query_params
)

city_data = city_input(data, start_date, end_date)
st.dataframe(city_data, width=1000)

mazandaran = [36.5700, 51.900]
zoom_level = 6.5

map(city_data.merge(coordinates, on="city", how="left")[["lat", "lon", "net", "tourist"]], mazandaran[0], mazandaran[1], zoom_level)

st.header("""انواع خوشه‌بندی""")
st.write("""
تا کنون راه‌حل‌های زیادی برای این مسئله ارائه شده است که از لحاظ معیار تشخیص خوشه‌ها و نحوه‌ی انتخاب یک خوشه، با یک‌دیگر تفاوت بسیاری دارند. ما برای کامل‌بودن بررسی خود از سه الگوریتم که اساس متفاوتی در کارکرد دارند استفاده کرده‌ایم. این الگوریتم‌ها عبارتند از:
""")
st.subheader("""مدل‌ مرکز گرا""")
st.write("""
در این روش برای هر دسته از داده‌های نزدیک به هم یک مرکز تعیین می‌شود که نشان‌دهنده نقطه‌ای است که خصوصیات اصلی دسته را دارد، بنابراین هر چه داده‌های ما از این مرکز دورتر شوند، با اطمینان کمتری در این دسته قرار می‌گیرند. روش k میانگین  (k-means) پرکاربردترین الگوریتم این دسته است که نقاط مرکز هر خوشه را براساس میانگین گرفتن از اعضایی که در هر مرحله در این دسته قرار دارند مشخص می‌کند. در واقع مرکز دسته نتیجه یک شورا بین اعضای فعلی دسته است.
در زیر می‌توانید نتایج شهرهای پرگردشگر را که از این روش به دست آمده است مشاهده کنید.
""")

first_city_population = aggregate_data(city_input(data, "1399-12-25", "1400-1-18"))
first_kmeans = kmeans(first_city_population, '1400')[["city", "cluster", "tourist", "tourist per capita"]]
second_city_population = aggregate_data(city_input(data, "1400-12-25", "1401-1-18"))
second_kmeans = kmeans(second_city_population, '1401')[["city", "cluster"]]
third_city_population = aggregate_data(city_input(data, "1401-12-25", "1402-1-18"))
third_kmeans = kmeans(third_city_population, '1402')[["city", "cluster"]]
st.write("""
مشاهده می‌کنید که در هر سال ما دو دسته شهر داریم که در بالا و سمت راست نمودار مشاهده شده‌اند و می‌توان از این شهرها به عنوان شهرهای گردشگری نام برد. دسته سمت راست شهرهایی است که پذیرای گردشگران بسیاری بوده‌اند و عموما شامل شهرهایی نام‌آشناست. دسته بالای نمودار کمی ناشناخته‌شده تر هستند. اگرچه این شهرها در نرخ گردشگر مقدار پایین‌تری داشته‌اند، اما تعداد گردشگر به ازای جمعیت آن‌ها مقدار قابل توجهی است.
""")
# Find cities that are in the same cluster for all three years and show them in a dataframe along their clusters
first_kmeans.columns = ["city", "1400 cluster", "tourist", "tourist per capita"]
second_kmeans.columns = ["city", "1401 cluster"]
third_kmeans.columns = ["city", "1402 cluster"]
kmeans_clusters = first_kmeans.merge(second_kmeans, on="city", how="inner").merge(third_kmeans, on="city", how="inner")
# Keep only cities that are in the same cluster for all three years
kmeans_clusters = kmeans_clusters[((kmeans_clusters["1400 cluster"] == 1) & (kmeans_clusters["1401 cluster"] == 2) & (kmeans_clusters["1402 cluster"] == 2)) | ((kmeans_clusters["1400 cluster"] == 2) & (kmeans_clusters["1401 cluster"] == 1) & (kmeans_clusters["1402 cluster"] == 1))]
kmeans_clusters = kmeans_clusters[kmeans_clusters["1401 cluster"] != 0]
kmeans_clusters = kmeans_clusters[kmeans_clusters["1402 cluster"] != 0]
# Drop 1400 cluster and 1401 cluster
kmeans_clusters.drop(columns=["1400 cluster", "1401 cluster"], inplace=True)
kmeans_clusters.columns = ["City", "Tourist", "Tourist Per Capita", "City Type"]
# Change 1 value in City Type to Most Popular
kmeans_clusters.loc[kmeans_clusters["City Type"] == 1, "City Type"] = "Based on Popularity"
kmeans_clusters.loc[kmeans_clusters["City Type"] == 2, "City Type"] = "Based on Tourist Per Capita"
st.dataframe(kmeans_clusters, width=1000)

st.subheader("مدل‌های مبتنی بر تراکم")
st.write("""
‫در این مدل، خوشه‌ها متناسب با ناحیه‌های متراکم نقاط در مجموعه داده مورد استفاده قرار می‌گیرد.
‫
""")
first_dbscan = dbscan(first_city_population, '1400')[["city", "cluster", "tourist", "tourist per capita"]]
second_dbscan = dbscan(second_city_population, '1401')[["city", "cluster"]]
third_dbscan = dbscan(third_city_population, '1402')[["city", "cluster"]]

first_dbscan.columns = ["city", "1400 cluster", "tourist", "tourist per capita"]
second_dbscan.columns = ["city", "1401 cluster"]
third_dbscan.columns = ["city", "1402 cluster"]

dbscan_clusters = first_dbscan.merge(second_dbscan, on="city", how="inner").merge(third_dbscan, on="city", how="inner")
dbscan_clusters = dbscan_clusters[dbscan_clusters["1400 cluster"] == dbscan_clusters["1401 cluster"]]
dbscan_clusters = dbscan_clusters[dbscan_clusters["1401 cluster"] == dbscan_clusters["1402 cluster"]]
dbscan_clusters = dbscan_clusters[dbscan_clusters["1401 cluster"] != 0]
dbscan_clusters.drop(columns=["1400 cluster", "1401 cluster", "1402 cluster"], inplace=True)
dbscan_clusters.columns = ["City", "Tourist", "Tourist Per Capita"]
st.dataframe(dbscan_clusters, width=1000)

st.subheader("مدل‌های مبتنی بر توزیع نقاط")
st.write("""
در این مدل، دسته‌ها با فرض پیروی از یک توزیع احتمالی مشخص می‌شوند.
از جمله الگوریتم‌های معروف ارائه شده در این مدل، الگوریتم بیشینه‌سازی امید ریاضی است.
""")
first_gmm = gmm(first_city_population, '1400')[["city", "cluster", "tourist", "tourist per capita"]]
second_gmm = gmm(second_city_population, '1401')[["city", "cluster"]]
third_gmm = gmm(third_city_population, '1402')[["city", "cluster"]]

first_gmm.columns = ["city", "1400 cluster", "tourist", "tourist per capita"]
second_gmm.columns = ["city", "1401 cluster"]
third_gmm.columns = ["city", "1402 cluster"]

gmm_clusters = first_gmm.merge(second_gmm, on="city", how="inner").merge(third_gmm, on="city", how="inner")
gmm_clusters = gmm_clusters[gmm_clusters["1400 cluster"] == gmm_clusters["1401 cluster"]]
gmm_clusters = gmm_clusters[gmm_clusters["1401 cluster"] == gmm_clusters["1402 cluster"]]
gmm_clusters = gmm_clusters[gmm_clusters["1401 cluster"] != 0]

gmm_clusters.drop(columns=["1400 cluster", "1401 cluster", "1402 cluster"], inplace=True)
gmm_clusters.columns = ["City", "Tourist", "Tourist Per Capita"]
st.dataframe(gmm_clusters, width=1000)

st.header("تست فرض")
st.markdown("""
در ابتدا برای تشخیص شهرهای گردشگری از شهرهای واسطه که محل استراحت یا گذر گردشگران هستند، شاخص میزان گردشگری را بر این اساس تعریف کردیم:

- \(X_{i}\): ورودی به هر شهر
  این متغیر بر اساس تفاوت تعداد ورودی و خروجی خودروها بوده که در صورت منفی بودن این تفاوت، آن را صفر در نظر گرفته‌ایم.
- \(P\): جمعیت هر شهر که بر اساس داده‌های سرشماری 1395 و برای بعضی شهرها از داده‌های 1385 استفاده شده است.

- \(\mu = 3.2\%\) 
- \(n\): بازه تعطیلات در هر سال

به ازای هر شهر یک نمونه \(n\) تایی از ورودی‌ها به آن در یک سال مشخص تهیه شده است.
ادعا می‌کنیم شهری گردشگری است که میانگین تعداد گردشگران آن به جمعیتش حداقل \(3.2\%\) باشد.
برای هر شهر در هر سال این آزمون با استفاده از جدول \(t\)-student بررسی شده و در نهایت به ازای هر شهر هایی که با این معیار گردشگری هستند را مشخص کرده‌ایم.

**فرضیه‌ها**:
- \(H_0\): میانگین \(\leq 0.032\)
- \(H_1\): میانگین \(> 0.032\)

در صورتی که 
\[
\frac{\bar{X} - \mu}{\frac{s}{\sqrt{n}}} < t_{(1-\alpha)}(n -1 )
\]
باشد فرض صفر رد شده و شهر گردشگری است.

برای تعیین برترین شهرها در این سه سال، از هر سال 10 شهر برتر انتخاب شده و شهرهایی به عنوان پرتقاضاترین شهرها معرفی شده‌اند که میانگین گردشگری آن‌ها در این سه سال بیشینه بوده است.

**نحوه محاسبه \(\mu\)**: 
میانگین نسبت گردشگری به جمعیت برای شهرهای بالای 100000 نفر در طی سه سال متوالی بررسی و میانگین آن‌ها به عنوان این معیار انتخاب شده است. داده‌های گلستان 1402 توسط سامانه ثبت نشده است؛ لذا با استفاده از رگرسیون به دنبال تخمین مناسبی برای این داده بودیم اما داده‌های حاصل نتایج درستی نداشتند. لذا برای این بخش به صورت روزانه میانگین داده‌های 1400 و 1401 برای هر شهر گلستان محاسبه و جایگزین شده است.
""")
hypothesis_test = load_test()
st.dataframe(hypothesis_test, width=1000)

st.header("نتیجه‌گیری نهایی")
st.write("""
براساس آنچه از آزمایش با چهار روش گفته شده در بالا به دست آمد، خوشه‌بندی با مدل مرکز‌ گرا بهترین نتیجه را به لحاظ تفسیر در اختیار ما قرار می‌دهد. بر این اساس می‌توان شهرها را به دو دسته تقسیم‌بندی کرد که دسته اول، بسیار گردشگرپذیر هست آمار توریست بالایی را به خود اختصاص داده‌اند و دسته دوم بیشترین پتانسیل را برای رشد دارند. این موضوع به دلیل استقبال بسیار زیاد گردشگران از این شهرها در ایام تعطیلات نوروز بوده است به طوری که در سیاه‌بیشته ما به ازای هر نفر از جمعیت شهر ۱۲۰۰ خودروی توریست داشته‌ایم.
""")
st.write("""
حال در پایین شهر دوآب را به عنوان یک نمونه از این شهرهای با پتانسیل بالا بررسی کرده و ابعاد مختلف آن را شرح می‌دهیم:
""")
st.subheader("دوآب کجاست")
st.write("""
از محور کرج ـ چالوس که به سمت چالوس حرکت کنید 7 کیلومتر بعد از مرزن‌آباد به دوراهى کجور مى‌رسید. در چند کیلومتری آن به سمت چالوس، به یک سه راهی می‌رسید که «‌دو آب کجور» ‌نام دارد‌. اگر از اینجا به سمت راست حرکت کنید، وارد جاده «دشت نظیر» می‌شوید 
اوایل جاده پوشیده از باغ هایی است که از سی چهل سال پیش احداث شده و در کنار جنگل دست کاشت کاج و سرو، سرسبزی و طراوتی به دشت و دره اطراف داده است 
""")
st.write("""
منطقه کجور دارای 69 روستا می باشد که به دهستان های پنجک رستاق ،زانوس رستاق و  شهر کجور و توابع کجور تقسیم بندی شده اند.
""")
st.subheader("تیتر خبری: صد فرصت شغلی با افتتاح دهکده گردشگری در کجور نوشهر ایجاد شد")
st.write("""
این منطقه با داشتن آبشار های متعدد ، دهکدده های گردشگری و فراهم کردن امکانات کمپ توانایی جذب مسافران زیادی را دارد.
در سال های اخیر  این ناحیه شاهد کلبه سازی های فراوانی بوده که امروزه آن را به یکی از جذاب ترین مکان های گردشگری برای استراحت در طبیعت بکر شمال کشور است.
دهکده زانوس  از معروف ترین دهکده گردشگری این منطقه است.
فصل بهار و تابستان برای سفر به این منطقه بسیار مناسب است.
""")
st.image('./images/doab-map.jpg')
st.write("""
در پایین تصویری از دهکده گردشگری زانوس را مشاهده می‌کنید که یکی از ده‌ها دهکده گردشگری واقع در این منطقه است و سالانه میزبان هزاران نفر گردشگر از شهرهای مختلف است.
""")
st.image('./images/village.jpg')
st.markdown("""
## چالش‌های پروژه

در این پروژه با چندین چالش مواجه شدیم که عبارتند از:

- **ثبت نشدن داده‌های تردد شمارها در ساعاتی از شبانه روز**: این موضوع می‌تواند منجر به از دست دادن اطلاعات مهم و یا نادرستی در تجزیه و تحلیل‌ها شود.

- **اختلاف در نگارش نام شهرها**: در فایل‌های دریافتی از سامانه، نام شهرها به شکل‌های مختلفی نوشته شده است، مانند "قائم شهر" و "قائمشهر"، که این امر باعث ایجاد مشکل در تشخیص و تجزیه و تحلیل داده‌ها شده است.

- **تشابه اسمی شهرها**: وجود شهرهایی با نام‌های یکسان در استان‌های مختلف می‌تواند موجب اشتباه شود.

- **در دسترس نبودن داده‌های گلستان برای سال 1402**: نبود داده‌ها می‌تواند تأثیر منفی بر کیفیت و دقت تجزیه و تحلیل‌ها بگذارد.

- **تفکیک نام‌های مشابه امامزاده‌ها**: در داده‌ها، دو محور منتهی به امامزاده‌هایی با نام "هاشم" وجود داشت که در دو استان متفاوت قرار دارند. تفکیک آن‌ها در ابتدا از طریق کدهای نوشته شده میسر نبود و مجبور به بررسی دستی شدیم.
            
- انتخاب معیار صحیح برای دسته‌بندی شهرها به شهرهای گردشگری و غیرگردشگری
""")
st.header("منابع")
st.markdown("""
- [داده‌های تردد شمار](https://141.ir/)
- [اتاقک](https://www.otaghak.com/blog/about-polur-village/)
- [وبلاگ گردشگری دوآب](https://www.jadidonline.com/story/12082009/frnk/galandorood)
- [تیتر خبری دوآب](https://roozno.com/fa/news/32826/%D8%A7%D8%B2-%D8%AF%D9%88%D8%A2%D8%A8-%DA%A9%D8%AC%D9%88%D8%B1-%D8%AA%D8%A7-%DA%AF%D9%84%D9%86%D8%AF%D8%B1%D9%88%D8%AF-%D8%B1%D8%B4%D8%AA%D9%87%E2%80%8C%D8%A7%DB%8C-%D8%A8%D8%A7-%D8%B1%D9%88%D8%B3%D8%AA%D8%A7%D9%87%D8%A7%DB%8C-%D8%B2%DB%8C%D8%A8%D8%A7)
- [جاذبه‌های گردشگری مازندران](http://evaz2mn.blogfa.com/category/15/-%D9%84%DB%8C%D8%B3%D8%AA-%DA%A9%D8%A7%D9%85%D9%84-%D8%A7%D8%B3%D8%A7%D9%85%DB%8C-%D8%B1%D9%88%D8%B3%D8%AA%D8%A7%D9%87%D8%A7%DB%8C-%D9%85%D9%86%D8%B7%D9%82%D9%87-%DA%A9%D8%AC%D9%88%D8%B1-)
""")
st.header("اعضای گروه")
st.markdown("""
- امیرمهدی سلیمانی‌فر (98101747)
- لیلی مطهری (99171214)
- رابعه پرهیزکاری (400109413)
""")