# Dogtor AI
Asimov Dog


Download Google drive file 
https://drive.google.com/drive/folders/1Cizpacxh4YzTHVSkSvmpXWVIK3nOJi7b?usp=sharing
and extract to the main directory under Dogtor-AI so the folder structure looks like:


Run command line in /Dorgot-AI/ folder
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WPv4F-Q_a8P_MtuTt0mpd5VRom1e4HIc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WPv4F-Q_a8P_MtuTt0mpd5VRom1e4HIc" -O DogZip.zip && rm -rf /tmp/cookies.txt

unzip DogZip.zip
sudo mv -v ./DogZip/* .
sudo rm DogZip.zip
rm DogZip -r
 
 Structure should look like this of files downloaded from drive
Dogtor-AI
    audio_files
    models


## Setup

Install libraries in requirements.txt. Use of a virtual env is suggested. On Linux/macOS:

python3 -m venv dogtor-ai-venv
source dogtor-ai-venv/bin/activate
pip install -r requirements.txt


## Usage

python listen_woof.py # starts the listener
