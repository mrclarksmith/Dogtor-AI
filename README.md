# Dogtor AI
## Overview
This Projects tries to provide a program that can detect a bark and the bark back using AI synthesided bark. 


## Downlaod DNN models folder
### Download Google drive file manually 
https://drive.google.com/drive/folders/1Cizpacxh4YzTHVSkSvmpXWVIK3nOJi7b?usp=sharing
and extract to the main directory under Dogtor-AI so the folder structure looks like:

### or 
### Run command line in /Dorgot-AI/ folder to download the google doc
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WPv4F-Q_a8P_MtuTt0mpd5VRom1e4HIc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WPv4F-Q_a8P_MtuTt0mpd5VRom1e4HIc" -O DogZip.zip && rm -rf /tmp/cookies.txt

unzip DogZip.zip
sudo mv -v ./DogZip/* .
sudo rm DogZip.zip
rm DogZip -r
 
 At the end the structure should look like this of files downloaded from drive:
- Dogtor-AI
    - models


## Setup

Install libraries in requirements.txt. Use of a virtual env is suggested. On Linux/macOS:

python3 -m venv dogtor-ai-venv
source dogtor-ai-venv/bin/activate
pip install -r requirements.txt


## Usage
python listen_woof.py # starts the the program and flask web server
For dashboard visit http://127.0.0.1:5000/ 

##  1. Description of code. 

### 1. Dog Bark idenfication
The processing of audio and detection happens inside check_woof.py file. 
Audio is captured and concatonated with .12 seconds of audio from previous recorded audio sample. This is done so audio bark is not split by the capture segment. Audio then is preproccessed then Image as classfied as dog or not dog score of 0 to 1  using BiT-M R101x1 model (https://tfhub.dev/google/bit/m-r101x1/ilsvrc2012_classification/1) with custom last layer of single output. 

### 2. The diagram shows the high level program flow and comnponents:

![Untitled Diagram drawio](https://user-images.githubusercontent.com/85537933/181687285-7e8fcf16-184e-4234-b384-18006418ef5a.png)

### 3. Image Classifier
![Image_classfier_model drawio](https://user-images.githubusercontent.com/85537933/181688253-36f4db77-0b63-40dc-b6e2-79474a9f96d7.png)

### 4. VAE neural network architecture coded following 'The Sound of AI' Youtube tutorial series by Valerio Velardo but modified for multiple TF modles loaded at the same time

![VAE_MEL_model drawio](https://user-images.githubusercontent.com/85537933/181694615-0d19abb4-3b9e-43c3-b964-65eb6662080f.png)

### 5. For Mel diagram to audio generation MelGAn was used
![MelGan](https://user-images.githubusercontent.com/85537933/181822016-58b68193-ff72-4eb0-be7f-0036c37a62a3.png)
source: https://github.com/seungwonpark/melgan

## 2. Data Sets:
Custom dog audio dataset was created from:
-1000 dog barsk from urbansound8k dataset
-60 hours of recorded audio barks 
-PetDogSoundEvent_Barking
Each bark audio file was split into multipe files with scilence removed. 
Sound was then listend to and any small dogs or wining pet sounds that didn't sound like a big dog or had too much background noise were removed. 
Total Dog files: 3,023.

Additionally for image classification trainig all but the dog sounds from the urbansound8k dataset were used.
During training for each epoch the audio files were randomly trimmed to training length. 
Total Non Dog files: 7,732.







