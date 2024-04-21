import subprocess
import sys
import os
import re

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components
import datetime
import plotly.express as px
import plotly.graph_objects as go
from persiantools.jdatetime import JalaliDate
import networkx as nx
from tqdm import tqdm
from pyvis.network import Network

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
def graph_formation(start_date, end_date):
    provinces = ['Gilan', 'Golestan', 'Mazandaran']

    # Read city-translation.csv
    city_translation = pd.read_csv("./helpers/city-translation.csv")
    road_translation = pd.read_csv("./helpers/road-translation.csv")

    G = nx.MultiDiGraph()
    start_date = JalaliDate(
        int(start_date.split('-')[0]),
        int(start_date.split('-')[1]),
        int(start_date.split('-')[2])
    )

    end_date = JalaliDate(
        int(end_date.split('-')[0]),
        int(end_date.split('-')[1]),
        int(end_date.split('-')[2])
    )

    classes = [1, 2]

    for province in provinces:
        files = os.listdir(f'{script_path}/graph-data/{province}')
        for file in files:
            name = re.sub(r'.csv', '', file)

            nodes = name
            edge = ''
            if '(' in name:
                name = name.split('(')
                nodes = name[0].strip()
                edge = name[1].strip()
                edge = edge[:-1]
            
            starting_node = nodes.split('-')[0].strip()
            ending_node = nodes.split('-')[1].strip()

            # English names
            if city_translation[city_translation['Persian'] == starting_node].shape[0] > 0:
                starting_node = city_translation[city_translation['Persian'] == starting_node]['English'].values[0]
            else:
                print(f"Starting node {starting_node} not found in city_translation.")
            if city_translation[city_translation['Persian'] == ending_node].shape[0] > 0:
                ending_node = city_translation[city_translation['Persian'] == ending_node]['English'].values[0]
            else:
                print(f"Ending node {ending_node} not found in city_translation.")

            df = pd.read_csv(f'{script_path}/graph-data/{province}/{file}')

            df['Start Date'] = df['Start Time'].apply(lambda x: JalaliDate(
            int(x.split(' ')[0].split('/')[0]),
            int(x.split(' ')[0].split('/')[1]),
            int(x.split(' ')[0].split('/')[2])
            ))

            df['End Date'] = df['End Time'].apply(lambda x: JalaliDate(
            int(x.split(' ')[0].split('/')[0]),
            int(x.split(' ')[0].split('/')[1]),
            int(x.split(' ')[0].split('/')[2])
            ))

            weight = 0
            # For all rows in df where the start date is between the start and end date
            for index, row in df.iterrows():
                if row['Start Date'] >= start_date and row['Start Date'] < end_date:
                    for c in classes:
                        weight += row[f'Number of Class {c} Vehicles']
            

            if 'قا' in starting_node:
                starting_node = 'Qaemshahr'
            if 'قا' in ending_node:
                ending_node = 'Qaemshahr'     

            if starting_node not in G:
                G.add_node(starting_node)
            if ending_node not in G:
                G.add_node(ending_node)

            if road_translation[road_translation['Persian'] == edge].shape[0] > 0:
                edge = road_translation[road_translation['Persian'] == edge]['English'].values[0]
            G.add_edge(starting_node, ending_node, label=f'{edge}({weight})', weight=np.log10(weight + 1))

    nt = Network(directed=True, notebook=False)
    nt.from_nx(G)

    for node in nt.nodes:
        incoming_edges = G.in_edges(node['id'])
        outgoing_edges = G.out_edges(node['id'])
        net = 0
        for edge in incoming_edges:
            # For each value
            for key, value in G[edge[0]][edge[1]].items():
                # Extract what's in value['label'] between the parentheses
                value = value['label']
                value = value[value.find('(')+1:value.find(')')]
                net += int(value)
        for edge in outgoing_edges:
            for key, value in G[edge[0]][edge[1]].items():
                value = value['label']
                value = value[value.find('(')+1:value.find(')')]
                net -= int(value)
        if net > 0:
            # Set size of node to be proportional to net
            node['size'] = 5 * np.log10(net + 10)
            node['color'] = 'green'
        elif net < 0:
            node['size'] = 5 * np.log10(-net + 10)
            node['color'] = 'red'
        else:
            node['size'] = 5
            node['color'] = 'blue'
        node['label'] = node['id'] + f'({net})'

    nt.toggle_physics(True)
    nt.force_atlas_2based(spring_length=150, central_gravity=0.003, gravity=-25, damping=0.9, overlap=1)
    HtmlFile = open("map.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height=700, width=700)

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
تلاش ما بر این بود که نتیجه را در قالب فرمتی تعاملی به نمایش درآوریم تا با افزودن المان‌هایی که کاربر امکان بررسی آن‌ها را دارد، میزان جذابیت گزارش بالا برود. به همین دلیل نتایج علاوه بر فرمت نوشتاری PDF به صورت یک صفحه وب تعامل نیز درآمده است که توصیه می‌کنیم برای بررسی نتایج در فرم کامل آن و آنچه ما دوست داریم به اشتراک بگذاریم، از آن استفاده کنید. لینک را می‌توانید از [اینجا](https://traffic-counter-app.streamlit.app) مشاهده کنید.
""")
st.write("""
‫شمال ایران با طبیعت بکر و آب و هوای دل‌پذیرش همواره یکی از مقاصد گردشگری محبوب بوده است. از رشت تا مازندران، از تالش تا آستارا، هر ناحیه‌ای در این منطقه دارای جذابیت‌های منحصر به فردی است که هر ساله گردشگران بسیاری را جذب می‌کند.
""")
st.image("./images/header-image.jpg", use_column_width=True)
st.write("""
در این پروژه، ما قصد داریم با بررسی داده های تردد شمار به  شناسایی و معرفی پرطرفدارترین شهرهای شمال ایران بپردازیم. این تحقیق به هدف ارائه اطلاعاتی کامل و مفید به گردشگران و علاقه‌مندان به سفر در این منطقه، و همچنین به عنوان یک منبع برای محققین و برنامه‌ریزان گردشگری و توسعه گردشگری ارائه می‌شود.
""")
st.header("""\"بزن بریم شمال\"""")
st.write("""‫همه ساله استان های شمالی کشور انتخاب بسیاری از ایرانیان در تعطیلات است اما بدون شک، بهار شمال چیز دیگری است! آغاز رونق طبیعت و سرسبزی دشت‌ها، در کنار آب و هوای معتدل و روزهای آفتابی، جذابیت خاصی به سفرها در این مناطق می‌بخشد. تعطیلات نوروز نیز بهانه‌ای عالی برای فرار از روزمرگی و پرداختن به سفرهای خانوادگی و دوستانه است. 
""")
st.write("""
با تحلیل تعداد ترددها که داده‌های تردد شمار در اختیار ما قرار داده است ، مشخص شده که شمال کشور حتی در دوران کرونا محبوبیت خود را به عنوان یک مقصد برای رفع خستگی سالانه حفظ کرده است.
همچنین پس از گذشت دوران طولانی اعمال محدودیت‌ها برای کنترل شیوع کرونا ، به نظر می‌رسد که جامعه به سمت زندگی پیش از کرونا برگشته است.
""")

cities_chart(data, ['Rasht', 'Sari', 'Gorgan'], 'Incoming Cars and Buses for Province Centers')


st.write("""
شاید فکر کنید که شهرهای با جمعیت تقریبا یکسان تجربه مشابهی از مسافرت نوروزی  ارائه می‌دهند. اگرباور شما چنین است توجه شما را به نمودار زیر جلب می‌کنیم:
""")

cities_chart(data, ['Astaneh', 'Asalem', 'Kiyakala'], 'Incoming Cars and Buses for Cities with Similar Population')

st.write("""
شهر‌های آستانه ، کیاکلا و اسالم به ترتیب با جمعیت 7166 و 8140 و 10720 توزیع گردشگری  یکسانی نداشته و مثال نقضی بر این باور هستند
و نشان می‌دهند مسافران مقصد خود را با معیار
های دیگر و به صورت هوشمندانه تری انتخاب می کنند.

در ادامه، ما به دنبال تحلیل و معرفی شهرهای از شمال کشور هستیم که در سه سال گذشته از محبوبیت بیشتری در میان مسافران  نوروزی برخوردار بوده‌اند، و از روش‌های مختلفی برای این تحلیل استفاده می‌کنیم. سپس، به بررسی مناطقی می‌پردازیم که پتانسیل گردشگری مناسبی دارند و شاهد توسعه و رشد این صنعت در این نواحی هستیم. 
‫
""")

st.header("داده‌های منسجم")
st.write("""
برای بررسی نمی‌توان این نکته را نادیده گرفت که برای نتیجه‌گیری درست نیاز به استفاده از یک منبع اطلاعاتی قابل اعتماد داریم، در غیراینصورت به نتیجه تلاش‌های ما اعتمادی نیست! به همین منظور با توجه به اینکه داده‌های خام تردد شمار پراکندگی به نسبت زیادی داشتند و میزان داده‌های ثبت نشده هم قابل چشم‌پوشی نبود، در مرحله اول ما پیش پردازش اولیه داده‌ها برای رسیدن به یک پایه صحیح را اولویت خود قرار دادیم.
""")
st.write("""
این پیش‌پردازش‌ها شامل ترکیب داده‌های محورهای یکسان بود که در قسمت‌های مختلفی از داده‌ها ثبت شده بودند، بررسی تک تک محورهای موجود و نامگذاری یکسان بود. به این ترتیب ما نقطه شروع خود را پیدا کردیم. نتیجه این تلاش، به وجود آمدن شبکه تعاملی زیر و نقشه داینامیک ما بود که بررسی‌های بعدی را با کمک آن انجام دادیم. اگر شما هم دوست دارید کمی با این داده‌ها کار کنید پیشنهاد ما، تغییر بازه زمانی زیر و مشاهده تغییرات تردد در محورها و بازدید از شهرهای شمالی است:
""")

start_date, end_date = st.select_slider(
    'Select the date range for analysis.',
    options=find_dates(),
    key='date',
    value=("1400-12-25", "1401-1-18"),
    on_change=update_query_params
)

city_data = city_input(data, start_date, end_date)
st.write("""
جدول ورودی و خروجی هر شهر و تعداد توریست‌ها (اتومبیل‌هایی که حداقل یک روز در شهر توقف داشته‌اند) در زیر آمده است:
""")
st.dataframe(city_data, width=1000)

mazandaran = [36.5700, 51.900]
zoom_level = 6.5
st.write("""
نقشه تعداد توریست‌های هر شهر که با ارتفاع ستون‌های آن نمایش داده شده است. شهر‌های با توریست زیاد به رنگ سبز پررنگ درآمده‌اند و شهرهایی که کمترین توریست را داشته‌اند قرمز شده‌اند.
""")
map(city_data.merge(coordinates, on="city", how="left")[["lat", "lon", "net", "tourist"]], mazandaran[0], mazandaran[1], zoom_level)
# Get absolute path of script.py
script_path = os.path.abspath(__file__)
# Remove script.py from the path
script_path = os.path.dirname(script_path)
# If folder graph-data does not exist, create it
if not os.path.exists(os.path.join(os.path.dirname(script_path), 'graph-data')):
    os.makedirs(os.path.join(os.path.dirname(script_path), 'graph-data'))
    # Unzip the graph-data.zip file
    with zipfile.ZipFile("graph-data.zip", 'r') as zip_ref:
        zip_ref.extractall(os.path.join(os.path.dirname(script_path), 'graph-data'))
st.write("""
گراف محورهای تردد که با افزایش ابعاد می‌توانید میزان تردد در هر یک را مشاهده کنید.
""")
graph_formation(start_date, end_date)

st.header("روش‌های بررسی")
st.write("""
حال که داده‌ها و ابزارهای تحلیل آن‌ها را به وجود آوردیم، می‌توانیم با استفاده از روش‌های آماری مانند خوشه‌بندی (clustering) و تست فرضیه (hypothesis testing) می‌تواند درک عمیق‌تری از داده‌ها و ترجیحات مسافران به دست آوریم. در ادامه، روش‌های مورد استفاده را به طور خلاصه توضیح می‌دهیم:
""")
st.subheader("روش خوشه‌بندی (Clustering)")
st.write("""
خوشه‌بندی یکی از تکنیک‌های یادگیری ماشین (شاخه‌ای از هوش مصنوعی) است که هدف آن دسته‌بندی داده‌ها بر اساس شباهت‌های میان آن‌ها است. در مورد تحلیل شهرهای شمال ایران، ما می‌توانیم از خوشه‌بندی برای گروه‌بندی شهرها بر اساس معیارهایی مانند تعداد گردشگران، نسبت گردشگر به جمعیت و نرخ ورود به شهرها استفاده کنیم. این تکنیک‌ها به ما کمک می‌کند تا شهرهایی که الگوهای مشابهی در جذب گردشگر دارند را شناسایی کنیم.
""")

st.subheader("تست فرض (Hypothesis Testing)")
st.write("""
تست فرض یک روش آماری است که برای ارزیابی صحت یک فرضیه بر اساس داده‌های موجود استفاده می‌شود. برای مثال، اگر می‌خواهیم بدانیم آیا تفاوت معنی‌داری بین تعداد گردشگران در دو شهر خاص در شمال ایران وجود دارد یا خیر، می‌توانیم از تست فرض استفاده کنیم. این روش به ما امکان می‌دهد تا با اطمینان بیشتری تصمیم‌گیری کنیم و مشخص کنیم که آیا تفاوت‌های مشاهده‌شده تصادفی هستند یا ناشی از عوامل بنیادین در جذب گردشگران.
""")

st.header("""نوبت پیداکردن خوشه‌هاست!""")
st.write("""
تا کنون راه‌حل‌های زیادی برای مشخص‌کردن دسته‌های مشابه از یک دسته داده ارائه شده است که از لحاظ معیار تشخیص خوشه‌ها و نحوه‌ی انتخاب یک خوشه، با یک‌دیگر تفاوت بسیاری دارند. ما برای کامل‌بودن بررسی خود از سه الگوریتم که اساس متفاوتی در کارکرد دارند استفاده کرده‌ایم. این الگوریتم‌ها عبارتند از:
""")

first_city_population = aggregate_data(city_input(data, "1399-12-25", "1400-1-18"))
second_city_population = aggregate_data(city_input(data, "1400-12-25", "1401-1-18"))
third_city_population = aggregate_data(city_input(data, "1401-12-25", "1402-1-18"))

st.subheader("مدل‌های مبتنی بر تراکم")
st.write("""
‫در این مدل، خوشه‌ها متناسب با ناحیه‌های متراکم نقاط در مجموعه داده مورد استفاده قرار می‌گیرد. در واقع معیار اصلی برای انتخاب یک دسته این است که نقاط زیادی کنار یکدیگر قرار داشته باشند. در مثال‌های روزمره زمانی که از شما می‌خواهند در یک جمعیت دسته‌ها را مشخص کنید، نگاه می‌کنید که در کدام نقاط تراکم افراد بیشتر است.
‫
""")
row1_1, row1_2 = st.columns([2, 1])
with row1_1:
    st.write("""
    الگوریتمی که ما در اینجا به عنوان نماینده این دسته انتخاب کرده‌ایم، DBSCAN نام دارد که تراکم دسته‌ها را با کشیدن یک شعاع دور هر نقطه داده انجام می‌دهد. نقاطی که در مرکز باشند و داده‌های زیادی درون دایره آن‌ها باشد نقاط مرکزی دسته و نقاطی که در مرز باشند و تعداد کمی نقطه در آن داشته باشیم نقاط مرزی دسته انتخاب می‌شوند.
    """)
    st.write("""
    در پایین نتایج برای بررسی در سه سال ۱۳۹۹، ۱۴۰۰ و ۱۴۰۱ آورده شده است. 
    """)
with row1_2:
    st.image("./images/dbscan.webp", use_column_width=True)
first_dbscan = dbscan(first_city_population, '1400')[["city", "cluster", "tourist", "tourist per capita"]]
second_dbscan = dbscan(second_city_population, '1401')[["city", "cluster", "tourist", "tourist per capita"]]
third_dbscan = dbscan(third_city_population, '1402')[["city", "cluster", "tourist", "tourist per capita"]]

first_dbscan.columns = ["city", "1400 cluster", "1400 tourist", "1400 tourist per capita"]
second_dbscan.columns = ["city", "1401 cluster", "1401 tourist", "1401 tourist per capita"]
third_dbscan.columns = ["city", "1402 cluster", "1402 tourist", "1402 tourist per capita"]

# Merge on city and sum tourist and tourist per capita columns
dbscan_clusters = first_dbscan.merge(second_dbscan, on="city", how="inner").merge(third_dbscan, on="city", how="inner")
dbscan_clusters["tourist"] = dbscan_clusters["1400 tourist"] + dbscan_clusters["1401 tourist"] + dbscan_clusters["1402 tourist"]
dbscan_clusters["tourist per capita"] = dbscan_clusters["1400 tourist per capita"] + dbscan_clusters["1401 tourist per capita"] + dbscan_clusters["1402 tourist per capita"]
dbscan_clusters.drop(columns=["1400 tourist", "1401 tourist", "1402 tourist", "1400 tourist per capita", "1401 tourist per capita", "1402 tourist per capita"], inplace=True)
dbscan_clusters = dbscan_clusters[dbscan_clusters["1400 cluster"] == dbscan_clusters["1401 cluster"]]
dbscan_clusters = dbscan_clusters[dbscan_clusters["1401 cluster"] == dbscan_clusters["1402 cluster"]]
dbscan_clusters = dbscan_clusters[dbscan_clusters["1401 cluster"] != 0]
dbscan_clusters.drop(columns=["1400 cluster", "1401 cluster", "1402 cluster"], inplace=True)
dbscan_clusters.columns = ["City", "Sum of Tourist", "Sum of Tourist Per Capita"]
dbscan_clusters = dbscan_clusters.sort_values(by="Sum of Tourist", ascending=False)
dbscan_clusters.index = np.arange(1, len(dbscan_clusters) + 1)
st.write("""
این الگوریتم شهرهایی که از نظر تعداد توریست به جمعیت و یا حجم پذیرش گردشگر تفاوت عمده‌ای با سایر شهرها را داشته‌اند به درستی انتخاب کرده است. براین اساس شهرهایی که در جدول زیر نمایش داده شده‌اند به عنوان شهرهای گردشگری در مجموع این سه سال انتخاب شده‌اند. اجماع میان سه سال نیز با اشتراک گرفتن شهرهایی صورت گرفته است که در هر سه سال در دسته گردشگری قرار گرفته‌اند.
""")
st.dataframe(dbscan_clusters, width=1000)

st.subheader("مدل‌های مبتنی بر توزیع نقاط")
st.write("""
در این مدل، دسته‌ها با فرض پیروی از یک توزیع احتمالی مشخص می‌شوند.
در اینجا ما فرضیه‌ای را ارائه می‌کنیم که داده‌ها از یک سری توزیع بنیادین به دست آمده‌اند و سعی ‌می‌کنیم این توزیع‌ها را پیدا کنیم. این توزیع‌ها در واقع نشان‌دهنده دسته‌ای هستند که داده متعلق به آن است.
""")
st.write("""
مدلی که در اینجا استفاده کردیم Gaussian Mixture Model نام دارد که فرض می‌کنیم داده‌ها از ترکیب تعدادی توزیع گوسی به دست آمده است و سعی می‌کند این توزیع‌ها را پیدا کند. در انتها احتمال اینکه هر داده از هر کدام از توزیع‌ها به دست آمده باشد محاسبه می‌شود و توزیعی که بیشترین احتمال تولید آن داده را دارد، همان دسته‌ای است که داده‌ در آن قرار خواهد گرفت. برای مثال فرض کنید ما تاسی داریم که نیمی از اوقات ۶ تولید می‌کند و در مقابل آن تاسی داریم که استاندارد است. حال اگر تاسی ۶ آمده باشند و از شما بپرسند این کدام تاس است، احتمالا شما تاس اول را انتخاب می‌کنید. این همان رویکرد اصلی الگوریتم GMM است.
""")
st.write("""
در پایین نتایج به دست آمده برای سه سال را با استفاده از این الگوریتم مشاهده می‌کنید.
""")
first_gmm = gmm(first_city_population, '1400')[["city", "cluster", "tourist", "tourist per capita"]]
second_gmm = gmm(second_city_population, '1401')[["city", "cluster", "tourist", "tourist per capita"]]
third_gmm = gmm(third_city_population, '1402')[["city", "cluster", "tourist", "tourist per capita"]]

first_gmm.columns = ["city", "1400 cluster", "1400 tourist", "1400 tourist per capita"]
second_gmm.columns = ["city", "1401 cluster", "1401 tourist", "1401 tourist per capita"]
third_gmm.columns = ["city", "1402 cluster", "1402 tourist", "1402 tourist per capita"]

gmm_clusters = first_gmm.merge(second_gmm, on="city", how="inner").merge(third_gmm, on="city", how="inner")
gmm_clusters["tourist"] = gmm_clusters["1400 tourist"] + gmm_clusters["1401 tourist"] + gmm_clusters["1402 tourist"]
gmm_clusters["tourist per capita"] = gmm_clusters["1400 tourist per capita"] + gmm_clusters["1401 tourist per capita"] + gmm_clusters["1402 tourist per capita"]
gmm_clusters.drop(columns=["1400 tourist", "1401 tourist", "1402 tourist", "1400 tourist per capita", "1401 tourist per capita", "1402 tourist per capita"], inplace=True)
# Find the cluster of City of Doab
cluster = gmm_clusters[gmm_clusters["city"] == "Doab"]["1400 cluster"].values[0]
gmm_clusters = gmm_clusters[gmm_clusters["1400 cluster"] == gmm_clusters["1401 cluster"]]
gmm_clusters = gmm_clusters[gmm_clusters["1401 cluster"] == gmm_clusters["1402 cluster"]]
gmm_clusters = gmm_clusters[gmm_clusters["1401 cluster"] != 0]

gmm_clusters.drop(columns=["1400 cluster", "1401 cluster", "1402 cluster"], inplace=True)
gmm_clusters.columns = ["City", "Sum of Tourist", "Sum of Tourist Per Capita"]
st.write("""
نتایج شهرهایی که گردشگری هستند در زیر آورده شده است. اگر چه GMM سه دسته مختلف تعیین کرده است که در آن دسته اول شهرهایی هستند که گردشگری نیستند و پس از آن نیز دو دسته داریم که با توجه به معیارها به دو دسته متفاوت تقسیم شده‌اند، اما به دلیل عدم پیداکردن تفسیر مناسب این دو دسته را با هم ترکیب کردیم. نتایج شهرهای گردشگری در زیر آورده شده است.
""")
st.dataframe(gmm_clusters, width=1000)

st.subheader("""مدل‌ مرکز گرا""")
st.write("""
در این روش برای هر دسته از داده‌های نزدیک به هم یک مرکز تعیین می‌شود که نشان‌دهنده نقطه‌ای است که خصوصیات اصلی دسته را دارد، بنابراین هر چه داده‌های ما از این مرکز دورتر شوند، با اطمینان کمتری در این دسته قرار می‌گیرند. روش k میانگین  (k-means) پرکاربردترین الگوریتم این دسته است که نقاط مرکز هر خوشه را براساس میانگین گرفتن از اعضایی که در هر مرحله در این دسته قرار دارند مشخص می‌کند. در واقع مرکز دسته نتیجه یک شورا بین اعضای فعلی دسته است.
در زیر می‌توانید نتایج شهرهای پرگردشگر را که از این روش به دست آمده است مشاهده کنید.
""")
first_kmeans = kmeans(first_city_population, '1400')[["city", "cluster", "tourist", "tourist per capita"]]
second_kmeans = kmeans(second_city_population, '1401')[["city", "cluster", "tourist", "tourist per capita"]]
third_kmeans = kmeans(third_city_population, '1402')[["city", "cluster", "tourist", "tourist per capita"]]
st.write("""
مشاهده می‌کنید که در هر سال ما دو دسته شهر داریم که در بالا و سمت راست نمودار مشاهده شده‌اند و می‌توان از این شهرها به عنوان شهرهای گردشگری نام برد. دسته سمت راست شهرهایی است که پذیرای گردشگران بسیاری بوده‌اند و عموما شامل شهرهایی نام‌آشناست. دسته بالای نمودار کمی ناشناخته‌شده تر هستند. اگرچه این شهرها در نرخ گردشگر مقدار پایین‌تری داشته‌اند، اما تعداد گردشگر به ازای جمعیت آن‌ها مقدار قابل توجهی است.
""")
# Find cities that are in the same cluster for all three years and show them in a dataframe along their clusters
first_kmeans.columns = ["city", "1400 cluster", "1400 tourist", "1400 tourist per capita"]
second_kmeans.columns = ["city", "1401 cluster", "1401 tourist", "1401 tourist per capita"]
third_kmeans.columns = ["city", "1402 cluster", "1402 tourist", "1402 tourist per capita"]
kmeans_clusters = first_kmeans.merge(second_kmeans, on="city", how="inner").merge(third_kmeans, on="city", how="inner")
kmeans_clusters["tourist"] = kmeans_clusters["1400 tourist"] + kmeans_clusters["1401 tourist"] + kmeans_clusters["1402 tourist"]
kmeans_clusters["tourist per capita"] = kmeans_clusters["1400 tourist per capita"] + kmeans_clusters["1401 tourist per capita"] + kmeans_clusters["1402 tourist per capita"]
kmeans_clusters.drop(columns=["1400 tourist", "1401 tourist", "1402 tourist", "1400 tourist per capita", "1401 tourist per capita", "1402 tourist per capita"], inplace=True)
# Keep only cities that are in the same cluster for all three years
kmeans_clusters = kmeans_clusters[((kmeans_clusters["1400 cluster"] == 1) & (kmeans_clusters["1401 cluster"] == 2) & (kmeans_clusters["1402 cluster"] == 2)) | ((kmeans_clusters["1400 cluster"] == 2) & (kmeans_clusters["1401 cluster"] == 1) & (kmeans_clusters["1402 cluster"] == 1))]
kmeans_clusters = kmeans_clusters[kmeans_clusters["1401 cluster"] != 0]
kmeans_clusters = kmeans_clusters[kmeans_clusters["1402 cluster"] != 0]
# Drop 1400 cluster and 1401 cluster
kmeans_clusters.drop(columns=["1400 cluster", "1401 cluster"], inplace=True)
kmeans_clusters.columns = ["City", "City Type", "Sum of Tourist", "Sum of Tourist Per Capita"]
# Change 1 value in City Type to Most Popular
kmeans_clusters.loc[kmeans_clusters["City Type"] == 1, "City Type"] = "Based on Popularity"
kmeans_clusters.loc[kmeans_clusters["City Type"] == 2, "City Type"] = "Based on Tourist Per Capita"
st.dataframe(kmeans_clusters, width=1000)
st.write("""
براساس این دسته‌بندی که مفهوم مناسبی نیز دارد، سه شهر ساری، رشت، و بابل از نظر میزان گردشگر در رتبه‌های اول قرار دارند. علاوه بر آن سه شهر گرزین خیل، دوآب و هزارچم نیز از نظر میزان گردشگر به جمعیت در رتبه‌های اول هستند. باقی شهرهایی که در مرزبندی الگوریتم خوشه‌بندی به عنوان شهر گردشگری قلمداد شده‌اند نیز قابل مشاهده است.
""")


st.header("تست فرض")
st.markdown("""
در ابتدا برای تشخیص شهر های گردشگری ازشهر های واسطه که محل استراحت یا گذر گردشگران هستند شاخص میزان گردشگری را بر این اساس تعریف کردیم : 
\n
$X_{i}$ = ورودی به هر شهر  
\n
این متغیر بر اساس تفاوت تعداد ورودی و خروجی خودرو ها بوده که در صورت منفی بودن این تفاوت آن را صفر در نظر گرفته ایم.\n
$P$ = جمعیت هر شهر که بر اساس داده هاش سرشماری 1395 و برای بعضی شهر ها از داده های1385 استفاده شده است.\n

$\mu$ = $3.2$ درصد
\n
$n$ =  بازه تعطیلات در هر سال  
\n
به ازای هر شهر یک نمونه n تایی از ورودی های به آن در یک سال مشخص تهیه شده است.
ادعا می  کنیم شهری گردشگری است که میانگین تعداد گردشگران آن به جمعیتش حداقل $3.2$ درصد باشد.
برای هر شهر در هر سال این آزمون با استفاده جدول $student-t$ بررسی شده و در نهایت به ازای هر شهر هایی که با این معیار گردشگری هستند را مشخص کرده ایم.
\n
$H_0$  : میانگین $\leq 0.032$\n
$H_1$ :  میانگین $>$ $0.032$ \n
درصورتی که 
$\\frac{\\bar{X} - \mu}{\\frac{s}{\sqrt{n}}}<t_{(\\alpha-1)}\\times (1-n)$  باشد فرض صفر رد شده و شهر گردشگری است. \n
برای تعیین برترین شهر ها در این سه سال از هر سال 10 تای برتر انتخاب شده و شهر هایی به عنوان پرتقاضاترین شهرها معرفی شده اند که میانگین گردشگری آن ها در این سه سال بیشینه بوده است.
""")
st.subheader("نحوه محاسبه $\mu$")
st.write("""
میانگین نسبت گردشگری به جمعیت برای شهر های بالای 100000 نفر در طی سه سال متوالی بررسی  و میانگین آن ها به عنوان این معیار انتخاب شده است . 
داده های گلستان 1402 توسط سامانه ثبت نشده است لذا با استفاده از رگرسیون به دنبال تخمین مناسبی  برای این داده بودیم اما داده های حاصل نتایج درستی نداشتند، لذا برای این بخش به صورت روزانه میانگین داده های 1400  و  1401 برای هر شهر گلستان محاسبه و جایگزین شده است.
""")
hypothesis_test = load_test()
st.dataframe(hypothesis_test, width=1000)

st.header("نتیجه‌گیری نهایی")
st.write("""
براساس آنچه از آزمایش با چهار روش گفته شده در بالا به دست آمد، خوشه‌بندی با مدل مرکز‌ گرا بهترین نتیجه را به لحاظ تفسیر در اختیار ما قرار می‌دهد. بر این اساس می‌توان شهرها را به دو دسته تقسیم‌بندی کرد که دسته اول، بسیار گردش‌گرپذیر هست آمار توریست بالایی را به خود اختصاص داده‌اند و دسته دوم بیشترین پتانسیل را برای رشد دارند. این موضوع به دلیل استقبال بسیار زیاد گردشگران از این شهرها در ایام تعطیلات نوروز بوده است به طوری که در سیاه‌بیشته ما به ازای هر نفر از جمعیت شهر ۱۲۰۰ خودروی توریست داشته‌ایم. به این ترتیب لیست شهرهای گردشگری همراه با دسته‌بندی آن‌ها در زیر آورده شده است:
""")
# unordered list
st.write("""
- شهرهای پرگردشگر: ساری، رشت، بابل، قائمشهر، آمل، فومن، محمودآباد، منجیل، امامزاده هاشم، سراوان، چالوس، نور
- شهرهای پرپتانسیل: گرزین‌خیل، گدوک، دوآب، هزارچم، سیاه‌بیشه
""")
st.write("""
حال می‌خواهیم شهرهایی که در بالا پیدا کردیم را به صورت دقیق‌تر بررسی کنیم و دلیل گردشگرهای زیاد را در دوران عید برای آن‌ها بیابیم.
""")
st.header("توریست، همه‌جا توریست!")
st.write("این قسمت مربوط به شهرهایی است که بیشترین گردشگر را داشته‌اند و در این دسته سرآمد بوده‌اند. نماینده‌هایی که برای این قسمت انتخاب کردیم، بابل و قائمشهر هستند. دلیل انتخابمان هم این بود که رشت و ساری به علت مرکز استان بودن احتمالا بیشتر برای شما آشنا باشند و پرداختن به آن‌ها کسل‌کننده است.")
st.subheader("بابل")
st.write("""
بابل یکی از قدیمی‌ترین شهرهای استان مازندران است. این شهر پس از رشت، پرجمعیت‌ترین شهرستان خطه شمال ایران به حساب می‌آید.
بهترین فصل برای تماشای بابل، بهار و مخصوصا اردی‌بهشت است.
""")
row2_1, row2_2 = st.columns([2, 1])
with row2_1:
    st.write("""
- **هفت آبشار**\n
هفت آبشار تیرکن، یکی از زیباترین آبشارهای شهرستان بابل است. هفت آبشار همانطور که از نامش پیداست، از مجاورت هفت آبشار تشکیل شده است و در مرز بابل و شهرستان سوادکوه قرار دارد.""")
with row2_2:
    st.image("./images/7abshar.jpg", use_column_width=True)
st.write("""
- **پل محمدحسن خان**\n
پل محمدحسن خان بزرگ ترین راه ارتباطی بخش های بندپی شرقی و بندپی غربی شهرستان بابل است که بنایی تاریخی در این شهر می باشد. این بنا پلی قدیمی مربوط به سده 12 هجری قمری است که بر روی رودخانه بابلرود شهرستان بابل استان مازندران واقع شده است.
""")
st.subheader("قائمشهر")
st.write("""
قائم‌شهر در استان مازندران قرارگرفته است. این شهر به خاطر داشتن خط ریلی و قرار گرفتن در مسیر تقاطع جاده فیروزکوه و هراز اهمیت زیادی دارد.  طبیعت زیبای جلگه قائم‌شهر، مکان‌های تاریخی و هوای معتدلش از دلایل سفر به این شهر هستند.""")
st.write("""
- **دریاچه گل پل؛ منظره تماشایی دوستی جنگل و دریاچه**\n""")
row3_1, row3_2 = st.columns([2, 1])
with row3_1:
    st.write("""
در قسمت جنوبی قائمشهر روستایی به نام برنجستانک قرار دارد. در اطراف روستا جنگل، سد و دریاچه‌ای زیبا وجود دارد. دریاچه گل پل از دو حوضچه بزرگ تشکیل‌شده و آب شیرین دریاچه باعث شده است که ماهیان زیادی در آن زندگی کنند و پرندگان مرتب سراغش را بگیرند.""")
with row3_2:
    st.image("./images/golpol.jpg", use_column_width=True)

st.write("""
    - **روستای ریکنده، دهکده‌ای شیرین و توریستی**\n""")
row4_1, row4_2 = st.columns([2, 1])
with row4_1:
    st.write("""
    روستای ریکنده در سال‌های اخیر به یکی از محبوب‌ترین مقصدهای گردشگری و یکی از جذاب‌ترین جاهای دیدنی قائمشهر تبدیل شده است.  این روستا هر ساله مقصد مسافران زیادی بوده و چشم اندازهای طبیعی زیادی دارد. در بخش هایی از این روستا می توانید، بافت روستایی و جنگلی را مشاهده کنید که البته برای اسکان نیز مناسب هستند. شاید یکی از جذابیت های این روستا وجود مزارع نیشکرها است. این روستا از دیرباز مکان بسیار مناسبی برای کشت نیشکر بوده و نقش مهمی نیز در رونق اقتصادی این منطقه دارد. بخش مهمی از درآمد مردمان این روستا از طریق کشت این محصول می گذرد.
    """)
with row4_2:
    st.image("./images/reykandeh.jpg", use_column_width=True)

st.header("نوبت به تحلیل فرصت‌هاست!")
st.write("""
از شهرهای پرگردشگر که عبور کنیم، شهرهایی وجود دارند که نسبت گردشگر به جمعیت آن‌ها بسیار بالاست که نمایشگر اقتصادی پویا و فرصت‌هایی است که درست بهره‌برداری شده‌اند. می‌توان علاوه بر تلاش برای بهبود این مناطق از آن‌ها الگوبرداری کرد تا بتوانیم شهرهای بیشتری مانند آن‌ها را توسعه دهیم. در اینجا دوآب و گرزین خیل را به عنوان نماینده‌هایی از شهرهایی که پتانسیل خود را به خوبی استفاده کرده‌اند، معرفی می‌کنیم.
""")
st.subheader("دوآب")
st.write("""
از محور کرج ـ چالوس که به سمت چالوس حرکت کنید 7 کیلومتر بعد از مرزن‌آباد به دوراهى کجور مى‌رسید. در چند کیلومتری آن به سمت چالوس، به یک سه راهی می‌رسید که «‌دو آب کجور» ‌نام دارد‌. اگر از اینجا به سمت راست حرکت کنید، وارد جاده «دشت نظیر» می‌شوید 
اوایل جاده پوشیده از باغ هایی است که از سی چهل سال پیش احداث شده و در کنار جنگل دست کاشت کاج و سرو، سرسبزی و طراوتی به دشت و دره اطراف داده است 
""")
st.write("""
منطقه کجور دارای 69 روستا می باشد که به دهستان های پنجک رستاق ،زانوس رستاق و  شهر کجور و توابع کجور تقسیم بندی شده اند.
""")
st.write("""
- **تیتر خبری: صد فرصت شغلی با افتتاح دهکده گردشگری در کجور نوشهر ایجاد شد.**""")
row5_1, row5_2 = st.columns([1, 1])
with row5_1:
    st.write("""
این منطقه با داشتن آبشار های متعدد ، دهکدده های گردشگری و فراهم کردن امکانات کمپ توانایی جذب مسافران زیادی را دارد.
در سال های اخیر  این ناحیه شاهد کلبه سازی های فراوانی بوده که امروزه آن را به یکی از جذاب ترین مکان های گردشگری برای استراحت در طبیعت بکر شمال کشور است.
دهکده زانوس  از معروف ترین دهکده گردشگری این منطقه است.
فصل بهار و تابستان برای سفر به این منطقه بسیار مناسب است.
""")
with row5_2:
    st.image('./images/village.jpg', use_column_width=True)
st.image('./images/doab-map.jpg', use_column_width=True)
st.subheader("گرزین خیل")

st.write("""
    گرزین خیل در نزدیکی منقطه‌ی گردشگری آلادشت است. شهر آلاشت، یکی از جاهای دیدنی شهرستان سوادکوه در استان مازندران است که به‌دلیل فاصله کم با تهران، طبیعت بکر و زیبا، آب و هوای مطبوع و دلپذیر و وجود جاذبه‌های طبیعی و تاریخی، کوچه‌های باریک و سنگ‌فرش و خانه‌های خشتی با سقف شیروانی‌ چوبی، در سال‌های اخیر مورد توجه گردشگران قرار گرفته است.
    آلاشت، بهشت تاریخی مازندران و منطقه‌ای رویایی و خیال انگیز با چشم‌اندازی از کوه‌های مه گرفته و باغ‌های میوه است که بر فراز ارتفاعات به شما امکان قدم زدن روی اقیانوسی از ابرها را می‌دهد؛ پدیده‌ای اعجاب‌انگیز که امکان ندارد تجربه آن را فراموش کنید. 
    """)
st.write("""
    این منطقه دارای تعداد زیادی کلبه و منطقه گردشگری است که با یک جستجوی ساده می‌توان تعداد زیادی از آن‌ها را پیدا کرد. این سکونت‌گاه‌ها یکی از دلایل مهم جذب گردشگران به این منقطه و رونق اقتصاد آن هستند.
    """)
row6_1, row6_2 = st.columns([1, 1])
with row6_1:
    st.image('./images/alasht.jpg', use_column_width=True)
with row6_2:
    st.image('./images/alasht-map.png', use_column_width=True)
st.image('./images/cottage.png', use_column_width=True)
st.markdown("""
## چالش‌های پروژه

در این پروژه با چندین چالش مواجه شدیم که عبارتند از:

- **ثبت نشدن داده‌های تردد شمارها در ساعاتی از شبانه روز**: این موضوع می‌تواند منجر به از دست دادن اطلاعات مهم و یا نادرستی در تجزیه و تحلیل‌ها شود.

- **اختلاف در نگارش نام شهرها**: در فایل‌های دریافتی از سامانه، نام شهرها به شکل‌های مختلفی نوشته شده است، مانند "قائم شهر" و "قائمشهر"، و حتی یکسان نبودن Unicode مربوط به حروف با وجود نمایش یکسان که با چشم قابل تشخیص نیست. این امر باعث ایجاد مشکل در تشخیص و تجزیه و تحلیل داده‌ها شده است.

- **تشابه اسمی شهرها**: وجود شهرهایی با نام‌های یکسان در استان‌های مختلف می‌تواند موجب اشتباه شود.

- **در دسترس نبودن داده‌های گلستان برای سال 1402**: نبود داده‌ها می‌تواند تأثیر منفی بر کیفیت و دقت تجزیه و تحلیل‌ها بگذارد.

- **تفکیک نام‌های مشابه امامزاده‌ها**: در داده‌ها، دو محور منتهی به امامزاده‌هایی با نام "هاشم" وجود داشت که در دو استان متفاوت قرار دارند. تفکیک آن‌ها در ابتدا از طریق کدهای نوشته شده میسر نبود و مجبور به بررسی دستی شدیم.
            
- **انتخاب معیار صحیح برای دسته‌بندی شهرها به شهرهای گردشگری و غیرگردشگری:** با توجه به وجود دسته‌بندی‌های مختلف، انتخاب بهترین روش نیازمند بررسی بیشتر هر روش و تشخیص نقاط قوت و ضعف هر کدام بود.

- **پیداکردن دلیل گردشگر بالا در شهرهای مختلف:** با توجه به اینکه در داده‌ها نام محور‌ها ذکر شده نه نام مقاصد، پیداکردن دلیل گردشگر بالا نیازمند تحقیق و بررسی بیشتر بود. این مورد را به خصوص می‌توان در مورد شهرهایی که نسبت گردشگر به جمعیت زیادی داشتند، مانند گرزین خیل و دوآب، مشاهده کرد.
""")
st.header("منابع")
st.markdown("""
- [داده‌های تردد شمار](https://141.ir/)
- [اتاقک](https://www.otaghak.com/blog/about-polur-village/)
- [وبلاگ گردشگری دوآب](https://www.jadidonline.com/story/12082009/frnk/galandorood)
- [تیتر خبری دوآب](https://roozno.com/fa/news/32826/%D8%A7%D8%B2-%D8%AF%D9%88%D8%A2%D8%A8-%DA%A9%D8%AC%D9%88%D8%B1-%D8%AA%D8%A7-%DA%AF%D9%84%D9%86%D8%AF%D8%B1%D9%88%D8%AF-%D8%B1%D8%B4%D8%AA%D9%87%E2%80%8C%D8%A7%DB%8C-%D8%A8%D8%A7-%D8%B1%D9%88%D8%B3%D8%AA%D8%A7%D9%87%D8%A7%DB%8C-%D8%B2%DB%8C%D8%A8%D8%A7)
- [جاذبه‌های گردشگری مازندران](http://evaz2mn.blogfa.com/category/15/-%D9%84%DB%8C%D8%B3%D8%AA-%DA%A9%D8%A7%D9%85%D9%84-%D8%A7%D8%B3%D8%A7%D9%85%DB%8C-%D8%B1%D9%88%D8%B3%D8%AA%D8%A7%D9%87%D8%A7%DB%8C-%D9%85%D9%86%D8%B7%D9%82%D9%87-%DA%A9%D8%AC%D9%88%D8%B1-)\n
و موارد دیگر
""")
st.header("اعضای گروه")
st.markdown("""
- امیرمهدی سلیمانی‌فر (98101747)
- لیلی مطهری (99171214)
- رابعه پرهیزکاری (400109413)
""")