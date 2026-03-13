from dotenv import load_dotenv
from googleapiclient.discovery import build
import os
from langchain_core.runnables import RunnableLambda

load_dotenv()

youtube = build('youtube', 'v3' , developerKey=os.getenv("YOUTUBE_API_KEY"))

query = "supabase auth python"


def youtube_search(query):
    search_ = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=2,
            publishedAfter="2025-01-01T00:00:00Z"
    )

    response = search_.execute()

    videos_metadata = []
    for item in response["items"]:
        videos_metadata.append({
                "videoId": item["id"]["videoId"],          # ✅ Video ID
                "title": item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"],
                "link": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "publishedAt": item["snippet"]["publishedAt"],     
        })

    videos_metadata

