from python_dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

youtube = build('youtube', 'v3')
