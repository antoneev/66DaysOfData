import wget as wget
import zipfile
import os

def main():
    url = "https://dl.dropbox.com/s/5k9fsefk8yt3y01/files.zip"  # change www to dl and remove dl from behind zip to download
    save_path = "files/"
    filename = wget.download(url)

    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall()

    if os.path.exists("files.zip"):
        os.remove("files.zip")

if __name__ == '__main__':
    main()