import os
import time
from typing import Iterable, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from googleapiclient.discovery import build
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.proxies import WebshareProxyConfig , GenericProxyConfig


load_dotenv()


def _get_youtube_client():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY is not set in the environment")

    return build("youtube", "v3", developerKey=api_key)


def search_youtube_videos(
    query: str,
    max_results: int = 3,
    published_after: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Search YouTube and return a list of simple video metadata dicts.
    """
    youtube = _get_youtube_client()

    params: Dict[str, Any] = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
    }

    if published_after:
        params["publishedAfter"] = published_after

    response = youtube.search().list(**params).execute()

    videos_metadata: List[Dict[str, Any]] = []
    for item in response.get("items", []):
        videos_metadata.append(
            {
                "videoId": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"],
                "link": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "publishedAt": item["snippet"]["publishedAt"],
            }
        )

    return videos_metadata


def fetch_video_transcripts(
    videos_metadata: Iterable[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Any]]:
    """
    For each video in `videos_metadata`, try to fetch a transcript in English.
    Fallback chain:
      1. Manual English transcript
      2. Auto-generated English transcript
      3. Translate any available transcript → English

    Uses a proxy to avoid YouTube blocking and sleeps between requests.
    Returns a list of (video_metadata, fetched_transcript) tuples.
    """
    # ytt_api = YouTubeTranscriptApi(
    #     proxy_config=WebshareProxyConfig(
    #         proxy_username="pozeshlt-rotate",
    #         proxy_password="ply89vk7jpcc", 
    #         ) 
    # )

    ytt_api = YouTubeTranscriptApi(
        proxy_config=GenericProxyConfig(
            http_url="http://85.198.96.242:3128",
            https_url="http://85.198.96.242:3128",
        )
    )


    def _fetch_one(video: Dict[str, Any]):
        try:
            time.sleep(1.5)  # Increased from 0.5s to be safe
            transcript_list = ytt_api.list(video_id=video["videoId"])

            try:
                fetched = transcript_list.find_transcript(["en"]).fetch()
            except NoTranscriptFound:
                try:
                    fetched = transcript_list.find_generated_transcript(["en"]).fetch()
                except NoTranscriptFound:
                    available = list(transcript_list)
                    if not available:
                        print(f"⚠ No transcripts at all for: {video['title']}")
                        return None
                    print(f"🔄 Translating [{available[0].language_code}] → en for: {video['title']}")
                    fetched = available[0].translate("en").fetch()

            print(f"✅ Got transcript: {video['title']}")
            return (video, fetched)

        except TranscriptsDisabled:
            print(f"🚫 Transcripts disabled: {video['title']}")
            return None
        except Exception as e:
            print(f"❌ Error ({video['title']}): {e}")
            return None

    # Run transcript fetches sequentially
    results = list(map(_fetch_one, videos_metadata))

    videos_with_transcripts = [r for r in results if r is not None]

    print(f"\n🎉 Total videos with transcripts: {len(videos_with_transcripts)}")
    return videos_with_transcripts


def _format_timestamps(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def build_docs_from_transcripts(
    videos_with_transcripts: Iterable[Tuple[Dict[str, Any], Any]],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[Document]:
    """
    Convert fetched YouTube transcripts into LangChain `Document` chunks.
    Mirrors the logic from the notebook's `docs` construction.
    """
    docs: List[Document] = []

    for video, fetched in videos_with_transcripts:
        snippets = getattr(fetched, "snippets", [])
        if not snippets:
            continue

        chunk_text = ""
        chunk_start = snippets[0].start

        for seg in snippets:
            chunk_text += " " + seg.text

            if len(chunk_text) >= chunk_size:
                timestamp_secs = int(chunk_start)

                docs.append(
                    Document(
                        page_content=chunk_text.strip(),
                        metadata={
                            "title": video["title"],
                            "channel": video["channel"],
                            "publishedAt": video["publishedAt"],
                            "timestamp": _format_timestamps(timestamp_secs),
                            "url_with_timestamp": (
                                f"{video['link']}&t={timestamp_secs}s"
                            ),
                        },
                    )
                )
                chunk_text = chunk_text[-overlap:]
                chunk_start = seg.start

        if chunk_text.strip():
            timestamp_secs = int(chunk_start)
            docs.append(
                Document(
                    page_content=chunk_text.strip(),
                    metadata={
                        "title": video["title"],
                        "channel": video["channel"],
                        "publishedAt": video["publishedAt"],
                        "timestamp": _format_timestamps(timestamp_secs),
                        "url_with_timestamp": (
                            f"{video['link']}&t={timestamp_secs}s"
                        ),
                    },
                )
            )

    return docs

