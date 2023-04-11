import streamlit as st
import pandas as pd
import plost

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('Youtube Comment Anaylsis')

st.sidebar.subheader('')
time_hist_color = st.sidebar.selectbox('Color by', ('temp_min', 'temp_max')) 

def show_instructions():
    st.write("To upload a YouTube video into URL StreamLab, follow these steps:")
    st.write("1. Go to the video you want to use in YouTube and copy the video URL.")
    st.write("2. Open URL StreamLab and click on the 'Add URL' button.")
    st.write("3. Paste the video URL into the 'Enter URL' field.")
    st.write("4. Click the 'Add' button to add the video to your playlist.")
    st.write("5. Your video should now appear in the URL StreamLab playlist.")

if st.button("How to upload a YouTube video into URL StreamLab"):
    show_instructions()
st.sidebar.subheader('Youtube API Key')
donut_theta = st.sidebar.selectbox('Enter Here', [1, 2, 3, 4, 5])

st.sidebar.subheader('Line chart parameters')
plot_data = st.sidebar.multiselect('Select data', ['temp_min', 'temp_max'], ['temp_min', 'temp_max'])
plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created by Nolan Nehring''')


# Row A
st.markdown('### Metrics')
col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")

# Row B
seattle_weather = pd.read_csv('https://raw.githubusercontent.com/tvst/plost/master/data/seattle-weather.csv', parse_dates=['date'])
stocks = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/stocks_toy.csv')

c1, c2 = st.columns((7,3))
with c1:
    st.markdown('### Heatmap')
    plost.time_hist(
    data=seattle_weather,
    date='date',
    x_unit='week',
    y_unit='day',
    color=time_hist_color,
    aggregate='median',
    legend=None,
    height=345,
    use_container_width=True)
with c2:
    st.markdown('### Donut chart')
    plost.donut_chart(
        data=stocks,
        theta=donut_theta,
        color='company',
        legend='bottom', 
        use_container_width=True)

# Row C
st.markdown('### Line chart')
st.line_chart(seattle_weather, x = 'date', y = plot_data, height = plot_height)
