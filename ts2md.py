import os
import argparse
import whisper
import warnings
from pathlib import Path
from PIL import Image
import piexif
try:
    import pyheif
    HAS_PYHEIF = True
except ImportError:
    HAS_PYHEIF = False
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata
import subprocess

# Suppress Whisper warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.tiff', '.bmp', '.gif'}

def is_video_file(file_path):
    return file_path.suffix.lower() in VIDEO_EXTENSIONS

def is_image_file(file_path):
    return file_path.suffix.lower() in IMAGE_EXTENSIONS

def transcribe_audio(model, file_path):
    print(f"Transcribing: {file_path}")
    result = model.transcribe(str(file_path))
    return result['text']

def exif_gps_to_decimal(gps):
    # gps: ((deg_num, deg_den), (min_num, min_den), (sec_num, sec_den))
    def to_deg(val):
        return float(val[0][0]) / float(val[0][1]) + \
               float(val[1][0]) / float(val[1][1]) / 60 + \
               float(val[2][0]) / float(val[2][1]) / 3600
    if gps and len(gps) == 2:
        lat = to_deg(gps[0])
        lon = to_deg(gps[1])
        return lat, lon
    return None, None

def extract_image_metadata(file_path):
    meta = {}
    try:
        if file_path.suffix.lower() == '.heic':
            if HAS_PYHEIF:
                heif_file = pyheif.read(str(file_path))
                meta['Image Size'] = f"{heif_file.size[0]}x{heif_file.size[1]}"
                # HEIC EXIF extraction is limited, skip for now
            else:
                meta['error'] = 'pyheif not installed; cannot extract HEIC metadata.'
        else:
            img = Image.open(file_path)
            meta['Image Size'] = f"{img.width}x{img.height}"
            exif_data = img.info.get('exif')
            if exif_data:
                exif_dict = piexif.load(exif_data)
                # Get creation time
                dt = exif_dict['Exif'].get(piexif.ExifIFD.DateTimeOriginal)
                if dt:
                    meta['Creation Date'] = dt.decode()
                # Get GPS
                gps = exif_dict.get('GPS')
                if gps and piexif.GPSIFD.GPSLatitude in gps and piexif.GPSIFD.GPSLongitude in gps:
                    lat_tuple = gps[piexif.GPSIFD.GPSLatitude]
                    lon_tuple = gps[piexif.GPSIFD.GPSLongitude]
                    lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef, b'N').decode()
                    lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef, b'E').decode()
                    lat, lon = exif_gps_to_decimal((lat_tuple, lon_tuple))
                    if lat_ref == 'S':
                        lat = -lat
                    if lon_ref == 'W':
                        lon = -lon
                    meta['GPS'] = f"{lat:.6f}, {lon:.6f}"
                # Get Make/Model
                make = exif_dict['0th'].get(piexif.ImageIFD.Make)
                if make:
                    meta['Make'] = make.decode(errors='ignore')
                model = exif_dict['0th'].get(piexif.ImageIFD.Model)
                if model:
                    meta['Model'] = model.decode(errors='ignore')
                # Description
                desc = exif_dict['0th'].get(piexif.ImageIFD.ImageDescription)
                if desc:
                    meta['Description'] = desc.decode(errors='ignore')
            # File info
            meta['File Name'] = file_path.name
            meta['File Size'] = f"{file_path.stat().st_size // 1024} KB"
            meta['MIME Type'] = Image.MIME.get(img.format, 'unknown')
            meta['Image Width'] = img.width
            meta['Image Height'] = img.height
    except Exception as e:
        meta['error'] = str(e)
    return meta

def extract_video_metadata(file_path):
    meta = {}
    try:
        # Use exiftool to extract metadata, including UserComment and XPComment
        result = subprocess.run([
            'exiftool',
            '-CreateDate',
            '-CreationDate',
            '-ModifyDate',
            '-TrackCreateDate',
            '-MediaCreateDate',
            '-FileModifyDate',
            '-FileName',
            '-FileSize',
            '-ImageWidth',
            '-ImageHeight',
            '-MIMEType',
            '-Make',
            '-Model',
            '-GPSLatitude',
            '-GPSLongitude',
            '-GPSPosition',
            '-Duration',
            '-Comment',
            '-Description',
            '-UserComment',
            '-XPComment',
            str(file_path)
        ], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                meta[key.strip()] = value.strip()
        # Prefer Description, then UserComment, then XPComment, then Comment
        desc = meta.get('Description') or meta.get('UserComment') or meta.get('XPComment') or meta.get('Comment')
        meta['Description'] = desc if desc else ''
    except Exception as e:
        meta['error'] = str(e)
    return meta

def main():
    parser = argparse.ArgumentParser(description="Transcribe video files to a Markdown file using Whisper and extract metadata.")
    parser.add_argument('-i', '--input', type=str, default='.', help='Input directory with media files (default: current directory)')
    parser.add_argument('-o', '--output', type=str, default='transcriptions.md', help='Output markdown file name (default: transcriptions.md)')
    parser.add_argument('--debug', action='store_true', help='Debug mode: only process one video and one image file')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_file = Path(args.output)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a valid directory.")
        return

    model = whisper.load_model("base")
    files = [f for f in input_dir.iterdir() if f.is_file() and (is_video_file(f) or is_image_file(f))]

    if not files:
        print(f"No media files found in '{input_dir}'.")
        return

    # Debug mode: only process one video and one image
    if args.debug:
        video = next((f for f in files if is_video_file(f)), None)
        image = next((f for f in files if is_image_file(f)), None)
        files = [f for f in [video, image] if f is not None]
        print(f"[DEBUG] Processing files: {[str(f) for f in files]}")
        for file in files:
            print(f"[DEBUG] File: {file.name}")
            if is_video_file(file):
                meta = extract_video_metadata(file)
                print("[DEBUG] Video Metadata:")
                for k, v in meta.items():
                    if k == 'Description' and not v:
                        print(f"  {k}: Not found")
                    else:
                        print(f"  {k}: {v}")
                print(f"[DEBUG] File System - Created: {file.stat().st_ctime}, Modified: {file.stat().st_mtime}")
            elif is_image_file(file):
                meta = extract_image_metadata(file)
                print("[DEBUG] Image Metadata:")
                for k, v in meta.items():
                    if k == 'Description' and not v:
                        print(f"  {k}: Not found")
                    else:
                        print(f"  {k}: {v}")
                print(f"[DEBUG] File System - Created: {file.stat().st_ctime}, Modified: {file.stat().st_mtime}")

    with output_file.open('w', encoding='utf-8') as md:
        for file in sorted(files):
            md.write(f"## {file.name}\n\n")
            # Extract metadata
            if is_video_file(file):
                meta = extract_video_metadata(file)
                md.write("**Metadata:**\n")
                fields = [
                    ('Creation Date', 'Creation Date'),
                    ('Description', 'Description'),
                    ('Duration', 'Video Length'),
                    ('Comment', 'Video Description'),
                    ('GPS', 'GPS'),
                    ('Make', 'Make'),
                    ('Model', 'Model'),
                    ('MIME Type', 'MIME Type'),
                    ('File Name', 'File Name'),
                    ('File Size', 'File Size'),
                    ('Image Width', 'Image Width'),
                    ('Image Height', 'Image Height'),
                ]
                for key, label in fields:
                    if key == 'Description':
                        if key in meta and meta[key]:
                            md.write(f"- {label}: {meta[key]}\n")
                        else:
                            md.write(f"- {label}: Not found\n")
                    else:
                        if key in meta:
                            md.write(f"- {label}: {meta[key]}\n")
                if not any(key in meta for key, _ in fields):
                    md.write("- No metadata found\n")
                text = transcribe_audio(model, file)
                md.write("\n**Transcription:**\n\n")
                md.write(f"{text}\n\n")
            elif is_image_file(file):
                meta = extract_image_metadata(file)
                md.write("**Metadata:**\n")
                fields = [
                    ('Creation Date', 'Creation Date'),
                    ('Description', 'Description'),
                    ('GPS', 'GPS'),
                    ('Make', 'Make'),
                    ('Model', 'Model'),
                    ('MIME Type', 'MIME Type'),
                    ('File Name', 'File Name'),
                    ('File Size', 'File Size'),
                    ('Image Width', 'Image Width'),
                    ('Image Height', 'Image Height'),
                ]
                for key, label in fields:
                    if key == 'Description':
                        if key in meta and meta[key]:
                            md.write(f"- {label}: {meta[key]}\n")
                        else:
                            md.write(f"- {label}: Not found\n")
                    else:
                        if key in meta:
                            md.write(f"- {label}: {meta[key]}\n")
                if not any(key in meta for key, _ in fields):
                    md.write("- No metadata found\n")
                md.write("\n**Notes:** _(add your notes here)_\n\n")

    print(f"Transcription and metadata extraction completed. Markdown saved to: {output_file}")

if __name__ == "__main__":
    main()