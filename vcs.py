import requests
import sys
from api import API_KEY_ASSEMBLY

#upload
upload_endpoint = "https://api.assemblyai.com/v2/upload"
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
headers = {'authorization': API_KEY_ASSEMBLY}
filename = sys.argv[1]

def upload(filename):
    def read_file(filename, chunk_size=5242880):
        with open(filename, 'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data

    upload_response = requests.post(upload_endpoint,
                        headers=headers,
                        data=read_file(filename))

    audio_url = upload_response.json()['upload_url']
    return audio_url

# transribe
def transcribe(audio_url):
    transcript_request = {"audio_url": audio_url}
    transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=headers)
    job_id = transcript_response.json()['id']
    return job_id

# poll
def poll(transcript_id):
    polling_endpoint = transcript_endpoint + '/' + transcript_id 
    polling_response = requests.get(polling_endpoint, headers=headers)
    return polling_response.json()

def get_transcription_result_url(audio_url):
    transcript_id = transcribe(audio_url)
    while True:
        data = poll(transcript_id)
        if data['status'] == 'completed':
            return data, None
        elif data['status'] == 'completed':
            return data, data['error']
        
audio_url = upload(filename)
data, error = get_transcription_result_url(audio_url)

print(data['text'])
        
# save transcript
