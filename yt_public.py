import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from utils.comments import process_comments, make_csv
import csv

def comment_threads(videoID, youtube, video_title):
    comments_list = []

    request = youtube.commentThreads().list(
        part='id,replies,snippet',
        videoId=videoID,
        maxResults=5,
    )
    response = request.execute()
    comments_list.extend(process_comments(response['items'], video_title))
    while response.get('nextPageToken', None):
        request = youtube.commentThreads().list(
            part='id,replies,snippet',
            videoId=videoID,
            pageToken=response['nextPageToken']
        )
        response = request.execute()
        comments_list.extend(process_comments(response['items'], video_title))
    print(f"Finished fetching comments for {videoID}, {len(comments_list)}")
    return comments_list

def playlist_comments(playlistID, youtube):
    comments_list = []

    request = youtube.playlistItems().list(
        part='id,snippet',
        playlistId=playlistID,
        maxResults=50,
    )
    response = request.execute()
    videos_list = response['items']
    while response.get('nextPageToken', None):
        request = youtube.playlistItems().list(
            part='id,snippet',
            playlistId=playlistID,
            pageToken=response['nextPageToken']
        )
        response = request.execute()
        videos_list.extend(response['items'])

    for video in videos_list:
        videoID = video['snippet']['resourceId']['videoId']
        video_title = video['snippet']['title']
        comments_list.extend(comment_threads(videoID, youtube, video_title))

    print(f"Finished processing comments for {len(videos_list)} videos in playlist {playlistID}")
    return comments_list

def main(playlistID, API_KEY):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = playlist_comments(playlistID, youtube)
    if not comments:
        print("No comments found in playlist.")
        return None
    else:
        csv_data = make_csv(comments, 'combined')
        return csv_data


if __name__ == "__main__":
    main()






