import csv
from datetime import datetime as dt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import html
from nltk.corpus import stopwords
from cleaner import (remove_repeated_characters, separate_digit_text, slang_look_up, remove_extra_space,remove_url, remove_alphanumerics,remove_words_start_with)

# Download required resources from nltk
nltk.download('punkt')
nltk.download('stopwords')

# Define lists of words to check for in comments
addition_words = ['add', 'insert', 'include']
subtraction_words = ['remove', 'subtract', 'delete', 'eliminate', 'exclude', 'get rid of']
question_words = ["what", "when", "where", "who", "why", "how", "question"]

# Create empty list to store comments
comments = []

# Get current date in 'dd-mm-yyyy' format
today = dt.today().strftime('%d-%m-%Y')

# Function to clean a comment
def clean_comment(comment):
    # Decode HTML-encoded characters
    comment = html.unescape(comment)
    # Remove URLs
    comment = remove_url(comment)
    
    # Remove repeated characters
    comment = remove_repeated_characters(comment)
    
    # Separate digits and text
    comment = separate_digit_text(comment)
    
    # Remove words starting with '@' or '#'
    comment = remove_words_start_with(comment, ['@', '#'])
    
    # Convert to lowercase
    comment = comment.lower()
    
    # Translate slang
    comment = slang_look_up(comment)
    
    # Remove extra space
    comment = remove_extra_space(comment)
    
    return comment

# Function to check if a comment contains any words from a given list
def check_words(comment, words_list):
    for word in words_list:
        if word in comment:
            return 1
    return 0

# Function to check if a comment is a question
def is_question(comment):
    return int(comment.strip().endswith('?'))

# Function to process comments from a list of response items
def process_comments(response_items, video_title, csv_output=False):
    comments = []
    for res in response_items:
        if 'topLevelComment' in res['snippet'].keys():
            comment = res['snippet']['topLevelComment']['snippet']
            comment['parentId'] = None
            comment['commentId'] = res['snippet']['topLevelComment']['id']
            comment['videoTitle'] = video_title  # add video title to comment dict
            comments.append(comment)
    
    # Clean comments
    cleaned_comments = [clean_comment(comment['textDisplay']) for comment in comments]
    
    # Perform sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(comment)['compound'] for comment in cleaned_comments]
    
    # Add sentiment scores to comments
    for i in range(len(comments)):
        comments[i]['sentiment'] = sentiment_scores[i]
    
    # Generate CSV file of comments
    if csv_output:
        make_csv(comments)
    print(f"Finished processing {len(comments)} comments")
    return comments

# Function to generate a CSV file of comments
def make_csv(comments, channelID=None):
    # Define header row for CSV file
    header = list(comments[0].keys()) + ['cleaned_comment', 'addition', 'subtraction', 'question', 'maybe question']
    
    # Create empty list to store unique comments (i.e. comments that have not already been added to CSV file)
    unique_comments = []

    # Define filename for CSV file
    if channelID:
        filename = f'comments_{channelID}.csv'
    else:
        filename = 'comments.csv'

    # Write comments

    with open(filename, 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for comment in comments:
            cleaned_comment = clean_comment(comment.get('textDisplay', ''))
            comment['cleaned_comment'] = cleaned_comment
            comment['addition'] = check_words(cleaned_comment, addition_words)
            comment['subtraction'] = check_words(cleaned_comment, subtraction_words)
            comment['maybe question'] = check_words(cleaned_comment, question_words)
            comment['question'] = is_question(cleaned_comment)

            # Check if comment already exists in the list
            if comment not in unique_comments:
                unique_comments.append(comment)

                # Write row with all keys in header, even if they are not present in comment
                row = {key: comment.get(key, '') for key in header}
                writer.writerow(row)

    with open(filename, 'r', encoding='utf8') as f:
        csv_data = f.read()

    return csv_data

