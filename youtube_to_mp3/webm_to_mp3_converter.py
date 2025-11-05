#!/usr/bin/env python3
"""
Simple WebM to MP3 Converter

This script converts .webm files to .mp3 using FFmpeg directly.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def check_ffmpeg():
    """Check if FFmpeg is available"""
    # First try the portable version in current directory
    portable_ffmpeg = Path(__file__).parent / "ffmpeg-8.0-essentials_build" / "bin" / "ffmpeg.exe"
    if portable_ffmpeg.exists():
        return str(portable_ffmpeg)
    
    # Then try system FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, check=True, timeout=10)
        return 'ffmpeg'
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def convert_webm_to_mp3(input_dir, quality="192", ffmpeg_path='ffmpeg'):
    """
    Convert all .webm files in a directory to .mp3
    
    Args:
        input_dir (str): Directory containing .webm files
        quality (str): MP3 quality in kbps
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Directory not found: {input_dir}")
        return 0
    
    webm_files = list(input_path.glob("*.webm"))
    if not webm_files:
        print("❌ No .webm files found")
        return 0
    
    print(f"🔄 Found {len(webm_files)} .webm files to convert...")
    
    converted = 0
    failed = 0
    
    for webm_file in webm_files:
        mp3_file = webm_file.with_suffix('.mp3')
        
        if mp3_file.exists():
            print(f"⏭️  Skipping {webm_file.name} (MP3 already exists)")
            continue
        
        try:
            # Use FFmpeg directly to convert
            cmd = [
                ffmpeg_path,
                '-i', str(webm_file),
                '-q:a', '0',  # Use VBR
                '-map', 'a',  # Only audio
                '-y',  # Overwrite output files
                str(mp3_file)
            ]
            
            print(f"🎵 Converting: {webm_file.name}")
            result = subprocess.run(cmd, capture_output=True, check=True)
            
            if mp3_file.exists():
                print(f"✅ Success: {mp3_file.name}")
                converted += 1
            else:
                print(f"❌ Failed: Output file not created")
                failed += 1
                
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error for {webm_file.name}: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ Unexpected error for {webm_file.name}: {e}")
            failed += 1
    
    print(f"\n📊 Conversion Summary:")
    print(f"✅ Successfully converted: {converted}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {input_path.absolute()}")
    
    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert WebM files to MP3")
    parser.add_argument('input_dir', help='Directory containing .webm files')
    parser.add_argument('--quality', choices=['128', '192', '256', '320'], 
                       default='192', help='MP3 quality in kbps (default: 192)')
    
    args = parser.parse_args()
    
    print("🎵 WebM to MP3 Converter")
    print("=" * 30)
    
    ffmpeg_path = check_ffmpeg()
    if not ffmpeg_path:
        print("❌ FFmpeg not found! Please install FFmpeg first.")
        print("Visit: https://ffmpeg.org/download.html")
        sys.exit(1)
    
    print(f"✅ Using FFmpeg: {ffmpeg_path}")
    
    converted = convert_webm_to_mp3(args.input_dir, args.quality, ffmpeg_path)
    
    if converted > 0:
        print(f"\n🎉 Successfully converted {converted} files!")
    else:
        print("\n😞 No files were converted.")


if __name__ == "__main__":
    main()