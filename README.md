YouTube Comments Sentiment Analysis Dashboard
This project provides an interactive Streamlit dashboard for sentiment analysis of YouTube comments over time. The dashboard displays the overall sentiment of comments, as well as a timeline of the count of positive, negative, and neutral comments.

Dependencies
Python 3.6 or higher
pandas
numpy
plotly
streamlit
google-auth
google-auth-oauthlib
google-auth-httplib2
google-api-python-client
Getting Started
Install the required packages:
bash
Copy code
pip install pandas numpy plotly streamlit google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
Run the Streamlit application:
bash
Copy code
streamlit run streamlit_app.py
Code Explanation
streamlit_app.py
This file contains the main code for the Streamlit dashboard.

Functions
get_credentials(): This function reads the client_secret.json file and returns the necessary credentials to access the YouTube Data API.
get_comments(): This function fetches the comments from a specific YouTube video using the YouTube Data API and returns a DataFrame containing the comments and their published dates.
sentiment_analysis(): This function performs sentiment analysis on the comments using the VADER sentiment analysis library and returns a DataFrame with the sentiment scores for each comment.
plot_timeline(): This function takes the DataFrame of comments with sentiment scores and plots the count of positive, negative, and neutral comments over time.
Overall_display_dashboard(): This function displays the Streamlit dashboard, including the overall sentiment score and the timeline plot.
Dashboard
The dashboard is divided into two sections:

Overall Sentiment: This section displays the overall average sentiment score of the comments.
Sentiment Categories Over Time: This section displays a line chart of the count of positive, negative, and neutral comments over time.
Usage
Enter the YouTube video URL and click "Analyze" to fetch comments and perform sentiment analysis. The dashboard will update with the overall sentiment score and the timeline plot.
