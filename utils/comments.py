import csv
import re
import string
from datetime import datetime as dt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import html
from nltk.corpus import stopwords
from cleaner import (remove_repeated_characters, separate_digit_text, slang_look_up, remove_extra_space,remove_url, remove_alphanumerics,remove_words_start_with)

nltk.download('punkt')
nltk.download('stopwords')
addition_words = ['add', 'plus', 'insert', 'include', 'append', 'make']
subtraction_words = ['remove', 'subtract', 'delete', 'eliminate', 'exclude', 'get rid of']
question_words = ["what", "when", "where", "who", "why", "how", "question"]
comments = []

today = dt.today().strftime('%d-%m-%Y')
# cd "C:\Users\Nolan Nehring\Documents\Comments"
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
def check_words(comment, words_list):
    for word in words_list:
        if word in comment:
            return 1
        print(word)
    return 0
def is_question(comment):
    return int(comment.strip().endswith('?'))
def process_comments(response_items, csv_output = False):
    for res in response_items:
        if 'topLevelComment' in res['snippet'].keys():
            comment = res['snippet']['topLevelComment']['snippet']
            comment['parentId'] = None
            comment['commentId'] = res['snippet']['topLevelComment']['id']
            comments.append(comment)
        # #Handle replies
        # if 'replies' in res.keys():
        #     for reply in res['replies']['comments']:
        #         comment = reply['snippet']
        #         comment['commentId'] = reply['id']
        #         comments.append(comment)
        # #Handle non-replies
        # else:
        #     comment = {}
        #     comment['snippet'] = res['snippet']['topLevelComment']['snippet']
        #     comment['snippet']['parentId'] = None
        #     comment['snippet']['commentId'] = res['snippet']['topLevelComment']['id']
        #     comments.append(comment['snippet'])
    
    # Clean comments
    cleaned_comments = [clean_comment(comment['textDisplay']) for comment in comments]
    
    # Perform sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(comment)['compound'] for comment in cleaned_comments]
    
    # Add sentiment scores to comments
    for i in range(len(comments)):
        comments[i]['sentiment'] = sentiment_scores[i]
    
    if csv_output:
        make_csv(comments)
    print(f"Finished processing {len(comments)} comments")
    return comments

def make_csv(comments, channelID=None):
    header = list(comments[0].keys()) + ['cleaned_comment', 'addition', 'subtraction', 'question', ]
    unique_comments = []

    if channelID:
        filename = f'comments_{channelID}_{today}.csv'
    else:
        filename = f'comments_{today}.csv'

    with open(filename, 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for comment in comments:
            cleaned_comment = clean_comment(comment['textDisplay'])
            comment['cleaned_comment'] = cleaned_comment
            comment['addition'] = check_words(cleaned_comment, addition_words)
            comment['subtraction'] = check_words(cleaned_comment, subtraction_words)
            comment['maybe question'] = check_words(cleaned_comment, question_words)
            comment['question'] = is_question(cleaned_comment)

            # Check if comment already exists in the list
            if comment not in unique_comments:
                unique_comments.append(comment)
                writer.writerow(comment)

    print(f"Finished writing {len(unique_comments)} unique comments to {filename}")

