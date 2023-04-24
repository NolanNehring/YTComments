# Import necessary libraries
from googleapiclient.discovery import build
from utils.comments import process_comments, make_csv

# Function to fetch comments for a specific video
def comment_threads(videoID, youtube, video_title):
    # Initialize an empty list to store the comments
    comments_list = []
    
    # Make the API request to fetch the comments
    request = youtube.commentThreads().list(
        part='id,replies,snippet',
        videoId=videoID,
        maxResults=5,  # Limit the number of comments to 5 per request
    )
    response = request.execute()
    comments_list.extend(process_comments(response['items'], video_title))
    
    # Fetch all remaining pages of comments
    while response.get('nextPageToken', None):
        request = youtube.commentThreads().list(
            part='id,replies,snippet',
            videoId=videoID,
            pageToken=response['nextPageToken']
        )
        response = request.execute()
        comments_list.extend(process_comments(response['items'], video_title))
    
    # Print a message to indicate completion of fetching comments for a video
    print(f"Finished fetching comments for {videoID}, {len(comments_list)}")
    return comments_list

# Function to fetch comments for all videos in a playlist
def playlist_comments(playlistID, youtube):
    # Initialize an empty list to store the comments
    comments_list = []
    
    # Make the API request to fetch the videos in the playlist
    request = youtube.playlistItems().list(
        part='id,snippet',
        playlistId=playlistID,
        maxResults=50,  # Limit the number of videos to 50 per request
    )
    response = request.execute()
    videos_list = response['items']
    
    # Fetch all remaining pages of videos in the playlist
    while response.get('nextPageToken', None):
        request = youtube.playlistItems().list(
            part='id,snippet',
            playlistId=playlistID,
            pageToken=response['nextPageToken']
        )
        response = request.execute()
        videos_list.extend(response['items'])
    
    # Fetch comments for each video in the playlist
    for video in videos_list:
        videoID = video['snippet']['resourceId']['videoId']
        video_title = video['snippet']['title']
        comments_list.extend(comment_threads(videoID, youtube, video_title))
    
    # Print a message to indicate completion of fetching comments for all videos in the playlist
    print(f"Finished processing comments for {len(videos_list)} videos in playlist {playlistID}")
    return comments_list

# Main function to fetch comments for a playlist and return the data as CSV
def main(playlistID, API_KEY):
    # Initialize the YouTube API client
    youtube = build("youtube", "v3", developerKey=API_KEY)
    
    # Fetch comments for the playlist
    comments = playlist_comments(playlistID, youtube)
    
    # If no comments are found, print a message and return None
    if not comments:
        print("No comments found in playlist.")
        return None
    
    # If comments are found, convert the data to CSV format and return it
    else:
        csv_data = make_csv(comments, playlistID)
        return csv_data

# Check if this file is being run as the main module
if __name__ == "__main__":
    main()
