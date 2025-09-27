import pyaudio
import websockets
import asyncio
import base64
import json
from api import API_KEY_ASSEMBLY

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()

device_index = None
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info.get('maxInputChannels') > 0:  # means it's a mic
        device_index = i
        print(f"Using input device {i}: {info.get('name')}")
        break

if device_index is None:
    raise OSError("No microphone input device found!")

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=device_index,  # ✅ auto-selected mic
    frames_per_buffer=FRAMES_PER_BUFFER
)

URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"

async def send_recieve():
    async with websockets.connect(
        URL,
        ping_timeout=20,
        ping_interval=5,
        extra_headers={"Authorization": API_KEY_ASSEMBLY}
    ) as _ws:
        await asyncio.sleep(0.1)
        session_begins = await _ws.recv()
        print(session_begins)
        print("Sending messages")

        async def send():
            while True:
                try:
                    data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                    data = base64.b64encode(data).decode("utf-8")
                    json_data = json.dumps({"audio_data": data})
                    await _ws.send(json_data)
                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break
                except Exception as e:
                    assert False, "Not a websocket 4008 error"
                await asyncio.sleep(0.01)

        async def receive():
            while True:
                try:
                    result_str = await _ws.recv()
                    result = json.loads(result_str)
                    prompt = result("text")
                    if prompt and result["message_type"] == "FinalTranscipt":
                        print("Me:", prompt)
                except websockets.execption.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break
                except Exception as e:
                    assert False, "Not a websocket 4008 error"

        send_result, receieve_result = await asyncio.gather(send(), receive())

asyncio.run(send_recieve())