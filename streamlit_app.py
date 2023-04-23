import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from urllib.parse import urlparse, parse_qs
from yt_public import main, make_csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import altair as alt
import os
import html
import webbrowser
import plotly.graph_objs as go
st.set_page_config(page_title="Youtube Comment Analysis", page_icon=":chart_with_upwards_trend:", layout="wide", initial_sidebar_state="expanded")

if os.path.exists("style.css"):
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
else:
    st.warning("The file style.css does not exist. Please make sure the file is in the correct directory.")
dirname = os.path.dirname(__file__)
css_filename = os.path.join(dirname, 'style.css')

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# streamlit run streamlit_app.py
#cd C:\Users\Nolan Nehring\Documents\YTComments-main
df = None
stemmer = PorterStemmer()
st.sidebar.header('Youtube Comment Anaylsis')
 
url = st.sidebar.text_input('Enter a YouTube URL')

# Extract the playlist ID from the URL
if url:
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    playlist_id = query_params.get('list', [''])[0]
    
    if parsed_url.query:
        query_params = parse_qs(parsed_url.query)
        if 'list' in query_params:
            playlist_id = query_params['list'][0]
            st.sidebar.write(f'Playlist ID: {playlist_id}')
        else:
            st.sidebar.write('Invalid URL: No playlist ID found')
    else:
        st.sidebar.write('Invalid URL: No query parameters found')

st.sidebar.subheader('Youtube API Key')
API_key = st.sidebar.text_input('Enter your YouTube API key')
def generate_data(playlist_id):
    data = main(playlist_id, API_key)
    # modify the data as needed
    return data
if st.sidebar.button('Generate Data'):
    csv_data = generate_data(playlist_id)
    df = pd.read_csv(StringIO(csv_data))
    df['textDisplay'] = df['textDisplay'].apply(html.unescape)
    df['textDisplay'] = df['textDisplay'].apply(lambda x: html.unescape(x).replace('<br>', ''))
st.sidebar.markdown('''
---
Created by Nolan Nehring''')
                    

def count_sentiment_categories(df):
# Compute sentiment categories
    positive_count = df[df['sentiment'] > 0.3]['sentiment'].count()
    negative_count = df[df['sentiment'] < -0.3]['sentiment'].count()
    neutral_count = df[(df['sentiment'] >= -0.3) & (df['sentiment'] <= 0.3)]['sentiment'].count()

# Create counts DataFrame
    counts = pd.DataFrame({
        'Category': ['Positive', 'Negative', 'Neutral'],
        'Count': [positive_count, negative_count, neutral_count]
    })

    # Specify the desired order of categories
    category_order = ['Positive', 'Neutral', 'Negative']

    # Create Altair donut chart
    chart = alt.Chart(counts).mark_arc(innerRadius=70, outerRadius=100).encode(
        theta=alt.Theta('Count', stack=True),
        color=alt.Color('Category', scale=alt.Scale(range=['#1f77b4', '#9467bd', '#d62728']),
                        sort=category_order),
        tooltip=['Category', 'Count']
    ).properties(
        width=300,
        height=300,
        title='Sentiment Categories'
    )

    # Show chart
    st.altair_chart(chart, use_container_width=True)

def display_comments(comments):
    with st.expander('Show/Hide comments'):
        st.table(comments[['cleaned_comment']])

def categorize_comments(df):
    categories = {
      
        'positive': df[(df['sentiment'] > 0.3) & (df['sentiment'] <= 1)],
        'negative': df[(df['sentiment'] >= -1) & (df['sentiment'] < -0.3)],
        'neutral': df[(df['sentiment'] >= -0.3) & (df['sentiment'] <= 0.3)],
    }
    return categories
def plot_timeline(df):
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df['week_start'] = df['publishedAt'].dt.to_period('W').apply(lambda r: r.start_time)

    df['sentiment_category'] = pd.cut(df['sentiment'], bins=[-1, -0.01, 0.01, 1], labels=['negative', 'neutral', 'positive'])

    weekly_counts = df.groupby(['week_start', 'sentiment_category']).size().reset_index(name='count')
    weekly_counts = weekly_counts.pivot_table(index='week_start', columns='sentiment_category', values='count', fill_value=0)

    fig = go.Figure()

    for sentiment in ['positive', 'neutral', 'negative']:
        fig.add_trace(go.Scatter(x=weekly_counts.index, y=weekly_counts[sentiment], mode='lines+markers', name=sentiment.capitalize()))

    fig.update_layout(title='Sentiment Categories Over Time', xaxis_title='Week', yaxis_title='Count')

    return fig

def preprocess_text(text):
    def remove_plural(token):
        if token.endswith('s'):
            return token[:-1]
        return token

    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    processed_tokens = [remove_plural(token) for token in filtered_tokens]
    return processed_tokens
def get_top_words(df, column='textDisplay', n=10):
    all_words = []

    for _, row in df.iterrows():
        text = row[column]
        tokens = preprocess_text(text)
        all_words.extend(tokens)

    word_freq = nltk.FreqDist(all_words)
    top_words = word_freq.most_common(n)

    return top_words
def generate_wordcloud(df, column='textDisplay'):
    all_words = []
    
    for _, row in df.iterrows():
        text = row[column]
        tokens = preprocess_text(text)
        all_words.extend(tokens)
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_words))
    
    
    return wordcloud
def display_comments(df, column_name, column_title, video_id):
    comments_df = df[df[column_name] == 1][['authorDisplayName', 'textDisplay', 'commentId']]
    if comments_df.shape[0] > 0:
        st.markdown(f'## {column_title} Comments')
        with st.expander('Show/Hide comments'):
            for i, row in comments_df.iterrows():
                comment = row['textDisplay']
                comment_id = row['commentId']
                author_name = row['authorDisplayName']
                button_label = "View on YouTube"
                button_url = f"https://www.youtube.com/watch?v={video_id}&lc={comment_id}"
                button_key = f"{column_name}_comment_{comment_id}_{video_id}"
                st.write(f"{author_name}: {comment}")
                st.button(button_label, key=button_key, on_click=lambda url=button_url: webbrowser.open_new_tab(url))
                st.write('---')
    else:
        st.write(f'No {column_title.lower()} comments')

##--------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------
avg_overall_score = None

def Overall_display_dashboard(df,avg_overall_sentiment):

    # Compute total sentiment score
    st.markdown("### Average Sentiment")
    st.markdown("# Overall Analysis")
    col1, col2 = st.columns(2)
    avg_sentiment = round(df["sentiment"].mean(), 3)
    
    col1.metric("Sentiment Score", f"{avg_sentiment:.3f}", )
    with col2:
        col2.write('## Sentiment Categories')
        count_sentiment_categories(df)
    st.markdown("## Sentiment Timeline")
    fig = plot_timeline(df)
    
    st.plotly_chart(fig)

    col21, col22 = st.columns(2)
    with col21:
        st.markdown("### Top Words")
        top_words = get_top_words(df, 'textDisplay', n=10)
        dfe = pd.DataFrame(top_words, columns=['Word', 'Count'])
        st.write(dfe.style.set_properties(**{'max-height': '1000px'}))
    with col22:
        st.markdown("### Word Cloud")
        wordcloud = generate_wordcloud(df, 'textDisplay')

        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    # Display the comments in separate windows
  
    if df[df['addition'] == 1].shape[0] > 0:
        st.markdown('## Addition Comments')

        with st.expander('Show/Hide comments'):
            comments = df[df['addition'] == 1][['authorDisplayName', 'textDisplay', 'videoId', 'commentId']]
            for i, row in comments.iterrows():
                comment = row['textDisplay']
                video_id = row['videoId']
                comment_id = row['commentId']
                author_name = row['authorDisplayName']
                button_label = "View on YouTube"
                button_url = f"https://www.youtube.com/watch?v={video_id}&lc={comment_id}"
                button_key = f"addition_comment_{i}"
                st.write(f"{author_name}: {comment}")
                st.button(button_label, key=button_key, on_click=lambda url=button_url: webbrowser.open_new_tab(url))
                st.write('---')
    else:
        st.write('No comments to show')

    if df[df['subtraction'] == 1].shape[0] > 0:
        st.markdown('## Minus Comments')

        with st.expander('Show/Hide comments'):
            comments = df[df['subtraction'] == 1][['authorDisplayName', 'textDisplay', 'videoId', 'commentId']]
            for i, row in comments.iterrows():
                comment = row['textDisplay']
                video_id = row['videoId']
                comment_id = row['commentId']
                author_name = row['authorDisplayName']
                button_label = "View on YouTube"
                button_url = f"https://www.youtube.com/watch?v={video_id}&lc={comment_id}"
                button_key = f"subtraction_comment_{i}"
                st.write(f"{author_name}: {comment}")
                st.button(button_label, key=button_key, on_click=lambda url=button_url: webbrowser.open_new_tab(url))
                st.write('---')
    else:
        st.write('No comments to show')

    if df[df['question'] == 1].shape[0] > 0:
        st.markdown('## Comments with Questions')

        with st.expander('Show/Hide comments'):
            comments = df[df['question'] == 1][['authorDisplayName', 'textDisplay', 'videoId', 'commentId']]
            for i, row in comments.iterrows():
                comment = row['textDisplay']
                video_id = row['videoId']
                comment_id = row['commentId']
                author_name = row['authorDisplayName']
                button_label = "View on YouTube"
                button_url = f"https://www.youtube.com/watch?v={video_id}&lc={comment_id}"
                button_key = f"question_comment_{i}"
                st.write(f"{author_name}: {comment}")
                st.button(button_label, key=button_key, on_click=lambda url=button_url: webbrowser.open_new_tab(url))
                st.write('---')
    else:
        st.write('No comments to show')
    st.markdown('---')

    
def display_dashboard(df,avg_overall_sentiment):
    # Compute total sentiment score
    col1, col2 = st.columns(2)
    avg_sentiment = round(df["sentiment"].mean(), 3)

    if avg_overall_sentiment is not None:
        percentage = ((avg_sentiment- avg_overall_sentiment)/ avg_overall_sentiment) * 100
        col1.metric("Sentiment Score", f"{avg_sentiment:.3f}",f"{percentage:.3f}")
    else:
        col1.metric("Sentiment Score", f"{avg_sentiment:.3f}")

    with col2:
        col2.write('## Sentiment Categories')
        count_sentiment_categories(df)
        
    col21, col22 = st.columns(2)
    with col21:
        st.markdown("### Top Words")
        top_words = get_top_words(df, 'textDisplay', n=10)
        st.write(pd.DataFrame(top_words, columns=['Word', 'Count']))
        
    with col22:
        st.markdown("### Word Cloud")
        wordcloud = generate_wordcloud(df, 'textDisplay')

        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    # Display the comments in separate windows
    display_comments(df, 'addition', 'Addition', video_id)
    display_comments(df, 'subtraction', 'Minus', video_id)
    display_comments(df, 'question', 'Question', video_id)

    st.markdown('---')
if df is not None:
    
    avg_overall_sentiment = round(df["sentiment"].mean(), 3)
    
    unique_video_ids = df['videoId'].unique()

    # Overall Analysis
    st.markdown("# Overall Analysis")
    
    Overall_display_dashboard(df, avg_overall_sentiment)
    

    # Analysis for each unique video ID
    unique_video_ids = df['videoId'].unique()
    for video_id in unique_video_ids:
        video_title = df[df['videoId'] == video_id]['videoTitle'].iloc[0]
        st.markdown(f"# Analysis for Video ID: {video_title}")
        video_df = df[df['videoId'] == video_id]
        display_dashboard(video_df, avg_overall_sentiment)

