# YouTube Playlist to MP3 Downloader & Converter

A complete solution for downloading YouTube playlists and converting them to high-quality MP3 files. This package includes two Python scripts that work together to download audio from YouTube and ensure proper MP3 conversion.

## 📦 What's Included

1. **`youtube_playlist_downloader.py`** - Main downloader for YouTube playlists/videos
2. **`webm_to_mp3_converter.py`** - Standalone converter for existing .webm files to MP3

## 🚀 Quick Start

### Prerequisites

- **Python 3.7+**
- **FFmpeg** (audio conversion tool)
- **uv** or **pip** (Python package manager)

### Installation

1. **Install Python dependencies:**
   ```bash
   uv pip install yt-dlp
   ```

2. **FFmpeg Setup:**
   
   The converter script comes with portable FFmpeg support. If you need to set it up:
   
   **Windows:**
   - Download from https://ffmpeg.org/download.html
   - Or use winget: `winget install "FFmpeg (Essentials Build)"`
   - Or the portable version will be auto-detected if placed in the script directory
   
   **macOS:**
   ```bash
   brew install ffmpeg
   ```
   
   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

## 📖 Usage Guide

### 1. Download YouTube Playlist as MP3

**Basic download (default quality: 192 kbps):**
```bash
uv run youtube_playlist_downloader.py "https://www.youtube.com/playlist?list=PLxxxxxxxx"
```

**Download to specific folder:**
```bash
uv run youtube_playlist_downloader.py "https://www.youtube.com/playlist?list=PLxxxxxxxx" my_music
```

**High-quality MP3 (320 kbps):**
```bash
uv run youtube_playlist_downloader.py "https://www.youtube.com/playlist?list=PLxxxxxxxx" downloads --quality 320
```

**Available quality options:**
- `128` - Standard quality (smaller files)
- `192` - High quality (default, balanced)
- `256` - Very high quality
- `320` - Maximum quality (larger files)

### 2. Convert Existing WebM Files to MP3

If you have .webm files that need conversion:

```bash
uv run webm_to_mp3_converter.py downloads
```

This will:
- Find all .webm files in the specified directory
- Convert them to MP3 format
- Skip files that already have MP3 versions
- Provide a conversion summary

### 3. Convert Existing Files Using Main Downloader

Alternatively, use the main downloader's built-in converter:

```bash
uv run youtube_playlist_downloader.py --convert-existing downloads
```

## 📋 Script Details

### YouTube Playlist Downloader (`youtube_playlist_downloader.py`)

**Features:**
- ✅ Downloads entire YouTube playlists or individual videos
- ✅ Automatic MP3 conversion with FFmpeg
- ✅ Configurable audio quality (128-320 kbps)
- ✅ Smart file naming: `01 - Song Title.mp3`
- ✅ Progress tracking and error handling
- ✅ Skips failed videos and continues downloading
- ✅ Built-in .webm to MP3 converter
- ✅ Auto-detects FFmpeg availability

**Command Line Options:**
```
youtube_playlist_downloader.py <playlist_url> [output_dir] [--quality QUALITY] [--convert-existing]

Arguments:
  playlist_url          YouTube playlist or video URL
  output_dir           Output directory (default: downloads)
  
Options:
  --quality {128,192,256,320}    MP3 quality in kbps (default: 192)
  --convert-existing             Convert existing .webm files to MP3
```

**Supported URL Formats:**
- Full playlist: `https://www.youtube.com/playlist?list=PLxxxxxxxx`
- Video in playlist: `https://www.youtube.com/watch?v=xxxxxxx&list=PLxxxxxxxx`
- Short URL: `https://youtu.be/xxxxxxx?list=PLxxxxxxxx`

### WebM to MP3 Converter (`webm_to_mp3_converter.py`)

**Features:**
- ✅ Batch converts all .webm files in a directory
- ✅ Uses portable or system FFmpeg
- ✅ High-quality VBR (Variable Bit Rate) encoding
- ✅ Skips already converted files
- ✅ Detailed progress and error reporting
- ✅ Standalone tool - works independently

**Command Line Options:**
```
webm_to_mp3_converter.py <input_dir> [--quality QUALITY]

Arguments:
  input_dir            Directory containing .webm files
  
Options:
  --quality {128,192,256,320}    MP3 quality in kbps (default: 192)
```

## 🔧 How It Works

### Download Process:
1. Script validates YouTube URL
2. Extracts playlist/video information
3. Downloads best audio quality available
4. FFmpeg converts audio to MP3 format
5. Files saved with playlist index and title

### Conversion Process:
1. Scans directory for .webm files
2. Checks if MP3 version already exists
3. Uses FFmpeg to extract and convert audio
4. Creates MP3 with high-quality VBR encoding
5. Reports success/failure for each file

## 📁 Output Format

Downloaded/converted files follow this naming pattern:
```
01 - Song Title.mp3
02 - Another Song.mp3
03 - Third Track.mp3
...
```

## 💡 Common Use Cases

### 1. Download a Music Playlist
```bash
uv run youtube_playlist_downloader.py "https://youtube.com/playlist?list=xxx" Music --quality 320
```

### 2. Fix Failed MP3 Conversions
If downloads completed but files are .webm instead of .mp3:
```bash
uv run webm_to_mp3_converter.py downloads
```

### 3. Batch Convert Old Downloads
Convert any folder of .webm files:
```bash
uv run webm_to_mp3_converter.py "path/to/webm/files"
```

### 4. Download with Standard Quality
Save bandwidth with good quality:
```bash
uv run youtube_playlist_downloader.py "https://youtube.com/playlist?list=xxx" --quality 192
```

## 🐛 Troubleshooting

### Issue: Files download as .webm instead of .mp3

**Cause:** FFmpeg is not installed or not in PATH

**Solution:**
1. Install FFmpeg (see Prerequisites)
2. Use the converter script: `uv run webm_to_mp3_converter.py downloads`

### Issue: "FFmpeg not found" error

**Solution:**
- **Windows:** Download portable FFmpeg and place in `ffmpeg-8.0-essentials_build/bin/` folder
- **macOS/Linux:** Install via package manager (`brew install ffmpeg` or `apt install ffmpeg`)
- Restart your terminal after installation

### Issue: Some videos fail to download

**Cause:** Videos may be private, deleted, or region-blocked

**Note:** The script continues downloading other videos in the playlist

### Issue: Download is very slow

**Cause:** Large playlists, slow internet, or YouTube throttling

**Tips:**
- Use lower quality setting (--quality 128)
- Download during off-peak hours
- Check your internet connection

## 📊 Performance Notes

- **Download speed:** Depends on internet connection and YouTube servers
- **Conversion speed:** Very fast with FFmpeg (typically < 1 second per song)
- **Disk space:** 320 kbps MP3 ≈ 2.5 MB per minute of audio
- **Batch processing:** Both scripts handle multiple files efficiently

## ⚖️ Legal & Ethics

- ✅ Use only for content you have permission to download
- ✅ Respect copyright laws in your jurisdiction
- ✅ Follow YouTube's Terms of Service
- ✅ Support artists by purchasing music when possible
- ❌ Do not distribute copyrighted content without permission

## 🔍 Technical Details

**Audio Quality:**
- **128 kbps:** ~1 MB per minute, decent quality
- **192 kbps:** ~1.4 MB per minute, high quality (recommended)
- **256 kbps:** ~1.9 MB per minute, very high quality
- **320 kbps:** ~2.4 MB per minute, maximum quality

**FFmpeg Settings:**
- Variable Bit Rate (VBR) for optimal quality/size ratio
- 44.1 kHz sample rate (standard for music)
- Mono/Stereo preserved from source
- ID3 tags preserved when available

## 🆘 Support

### Check FFmpeg Installation
```bash
ffmpeg -version
```

### Check Python Installation
```bash
python --version
```

### Update yt-dlp
```bash
uv pip install --upgrade yt-dlp
```

## 📝 Examples

### Example 1: Download Full Album
```bash
uv run youtube_playlist_downloader.py "https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf" "Albums/MyFavorite" --quality 320
```

### Example 2: Convert Old Downloads
```bash
# Convert all webm files in downloads folder
uv run webm_to_mp3_converter.py downloads

# Convert files in custom location
uv run webm_to_mp3_converter.py "C:/Users/YourName/Music/YouTube"
```

### Example 3: Quick Download
```bash
# Download with default settings (192 kbps, downloads folder)
uv run youtube_playlist_downloader.py "https://youtu.be/dQw4w9WgXcQ?list=PLxxxxxxxx"
```

## 🎯 Best Practices

1. **Test with small playlists first** to ensure everything works
2. **Use 192 kbps quality** for best balance of quality and file size
3. **Keep both .webm and .mp3 files** until you verify MP3 quality
4. **Organize downloads** into separate folders by artist/album
5. **Back up important downloads** to prevent data loss
6. **Update yt-dlp regularly** for compatibility with YouTube changes

## 📦 Dependencies

- **yt-dlp** - YouTube downloader (replaces deprecated youtube-dl)
- **FFmpeg** - Audio/video processing tool
- **Python 3.7+** - Programming language runtime

## 🎉 Features Summary

| Feature | Downloader | Converter |
|---------|-----------|-----------|
| Download from YouTube | ✅ | ❌ |
| Convert .webm to .mp3 | ✅ | ✅ |
| Batch processing | ✅ | ✅ |
| Quality selection | ✅ | ✅ |
| Progress tracking | ✅ | ✅ |
| Error handling | ✅ | ✅ |
| Portable FFmpeg support | ⚠️ | ✅ |
| Skip existing files | ✅ | ✅ |
| Playlist support | ✅ | ❌ |

---

**Made with ❤️ for music lovers who want offline access to their favorite YouTube content!**