#!/usr/bin/env python3
"""
YouTube Playlist MP3 Downloader

This script downloads all songs from a YouTube playlist and converts them to MP3 format.
Uses yt-dlp for downloading and ffmpeg for audio conversion.

Usage:
    python youtube_playlist_downloader.py <playlist_url> [output_directory]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import yt_dlp
from urllib.parse import urlparse


class YouTubePlaylistDownloader:
    def __init__(self, output_dir="downloads"):
        """
        Initialize the downloader with output directory
        
        Args:
            output_dir (str): Directory to save downloaded MP3 files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Check if FFmpeg is available
        if not self.check_ffmpeg():
            print("⚠️  Warning: FFmpeg not found. Installing yt-dlp[default] for built-in audio processing...")
            self.install_ytdlp_with_ffmpeg()
        
        # Configure yt-dlp options for MP3 download
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(self.output_dir / '%(playlist_index)02d - %(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'extractaudio': True,
            'audioformat': 'mp3',
            'ignoreerrors': True,  # Continue downloading even if some videos fail
            'no_warnings': False,
            'postprocessor_args': [
                '-ar', '44100'  # Set sample rate
            ],
        }
    
    def check_ffmpeg(self):
        """
        Check if FFmpeg is available on the system
        
        Returns:
            bool: True if FFmpeg is available, False otherwise
        """
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def install_ytdlp_with_ffmpeg(self):
        """
        Install yt-dlp with FFmpeg binaries included
        """
        try:
            print("Installing yt-dlp with embedded FFmpeg...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'yt-dlp[default]'], 
                         check=True, capture_output=True)
            print("✅ yt-dlp with FFmpeg support installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install yt-dlp with FFmpeg: {e}")
            print("Please install FFmpeg manually from https://ffmpeg.org/")
    
    def is_valid_youtube_url(self, url):
        """
        Validate if the URL is a valid YouTube playlist or video URL
        
        Args:
            url (str): URL to validate
            
        Returns:
            bool: True if valid YouTube URL, False otherwise
        """
        parsed = urlparse(url)
        return (
            parsed.netloc in ['www.youtube.com', 'youtube.com', 'youtu.be', 'm.youtube.com'] and
            ('playlist' in parsed.query or 'list=' in parsed.query or 'watch?v=' in parsed.query)
        )
    
    def download_playlist(self, playlist_url):
        """
        Download all songs from a YouTube playlist
        
        Args:
            playlist_url (str): YouTube playlist URL
            
        Returns:
            bool: True if download successful, False otherwise
        """
        if not self.is_valid_youtube_url(playlist_url):
            print(f"Error: Invalid YouTube URL: {playlist_url}")
            return False
        
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                print(f"Starting download from playlist: {playlist_url}")
                print(f"Output directory: {self.output_dir.absolute()}")
                
                # Extract playlist info first
                info = ydl.extract_info(playlist_url, download=False)
                if 'entries' in info:
                    total_videos = len([entry for entry in info['entries'] if entry])
                    print(f"Found {total_videos} videos in playlist: {info.get('title', 'Unknown')}")
                else:
                    print("Single video detected")
                
                # Download the playlist
                ydl.download([playlist_url])
                print("\n✅ Download completed successfully!")
                return True
                
        except yt_dlp.DownloadError as e:
            print(f"❌ Download error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def convert_webm_to_mp3(self):
        """
        Convert existing .webm files to .mp3 format
        
        Returns:
            int: Number of files converted
        """
        webm_files = list(self.output_dir.glob("*.webm"))
        if not webm_files:
            print("No .webm files found to convert")
            return 0
        
        print(f"Found {len(webm_files)} .webm files to convert to MP3...")
        converted = 0
        
        for webm_file in webm_files:
            mp3_file = webm_file.with_suffix('.mp3')
            if mp3_file.exists():
                print(f"⏭️  Skipping {webm_file.name} (MP3 already exists)")
                continue
            
            try:
                with yt_dlp.YoutubeDL({
                    'outtmpl': str(mp3_file.with_suffix('')),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                }) as ydl:
                    # Extract audio from the webm file
                    info = {
                        'filepath': str(webm_file),
                        'ext': 'webm'
                    }
                    ydl._make_archive_id = lambda info_dict: None  # Disable archive
                    ydl.process_info(info)
                
                print(f"✅ Converted: {webm_file.name} → {mp3_file.name}")
                converted += 1
                
                # Optionally remove the original .webm file
                # webm_file.unlink()  # Uncomment to delete .webm files after conversion
                
            except Exception as e:
                print(f"❌ Failed to convert {webm_file.name}: {e}")
        
        return converted
    
    def list_downloaded_files(self):
        """List all MP3 files in the output directory"""
        mp3_files = list(self.output_dir.glob("*.mp3"))
        if mp3_files:
            print(f"\n📁 Downloaded files ({len(mp3_files)} total):")
            for file in sorted(mp3_files):
                file_size = file.stat().st_size / (1024 * 1024)  # Convert to MB
                print(f"  • {file.name} ({file_size:.1f} MB)")
        else:
            print("\n📁 No MP3 files found in output directory")


def main():
    """Main function to handle command line arguments and run the downloader"""
    parser = argparse.ArgumentParser(
        description="Download YouTube playlist as MP3 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://www.youtube.com/playlist?list=PLxxxxxxxx"
  %(prog)s "https://www.youtube.com/playlist?list=PLxxxxxxxx" my_music
  %(prog)s "https://www.youtube.com/watch?v=xxxxxxx&list=PLxxxxxxxx" downloads
        """
    )
    
    parser.add_argument(
        'playlist_url',
        help='YouTube playlist URL or video URL with playlist'
    )
    
    parser.add_argument(
        'output_dir',
        nargs='?',
        default='downloads',
        help='Output directory for downloaded MP3 files (default: downloads)'
    )
    
    parser.add_argument(
        '--quality',
        choices=['128', '192', '256', '320'],
        default='192',
        help='MP3 quality in kbps (default: 192)'
    )
    
    parser.add_argument(
        '--convert-existing',
        action='store_true',
        help='Convert existing .webm files in output directory to MP3'
    )
    
    args = parser.parse_args()
    
    # Create downloader instance
    downloader = YouTubePlaylistDownloader(args.output_dir)
    
    # Update quality setting if specified
    downloader.ydl_opts['postprocessors'][0]['preferredquality'] = args.quality
    
    print("🎵 YouTube Playlist MP3 Downloader")
    print("=" * 40)
    
    # Handle conversion of existing files
    if args.convert_existing:
        print("🔄 Converting existing .webm files to MP3...")
        converted = downloader.convert_webm_to_mp3()
        if converted > 0:
            print(f"✅ Successfully converted {converted} files to MP3!")
        downloader.list_downloaded_files()
        return
    
    # Download the playlist
    success = downloader.download_playlist(args.playlist_url)
    
    if success:
        # List downloaded files
        downloader.list_downloaded_files()
        print(f"\n🎉 All done! Files saved to: {downloader.output_dir.absolute()}")
    else:
        print("\n❌ Download failed. Please check the URL and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()