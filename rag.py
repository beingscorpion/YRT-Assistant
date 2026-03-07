from dotenv import load_dotenv
from googleapiclient.discovery import build
import os

load_dotenv()

youtube = build('youtube', 'v3' , developerKey=os.getenv("YOUTUBE_API_KEY"))

query = "supabase auth python"


