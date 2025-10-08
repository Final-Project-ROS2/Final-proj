import pyaudio
import websockets
import asyncio
import base64
import json
from api import API_KEY_ASSEMBLY
from custom_interfaces.srv import GetTranscript

import rclpy
from rclpy.node import Node


FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


class ASRService(Node):

    def __init__(self):
        super().__init__('asr_service')
        self.srv = self.create_service(GetTranscript, 'get_transcript', self.asr_callback)
        self.get_logger().info('ASR Service initialized')

    def asr_callback(self, request, response):
        """Service callback to perform ASR"""
        self.get_logger().info('Incoming request: duration=%d seconds' % request.duration)
        
        try:
            transcript = asyncio.run(self.run_asr(request.duration))
            response.transcript = transcript
            response.success = True
        except Exception as e:
            self.get_logger().error(f'ASR Error: {str(e)}')
            response.transcript = ""
            response.success = False
        
        return response

    async def run_asr(self, duration_seconds):
        """Run ASR for specified duration or until silence"""
        p = pyaudio.PyAudio()

        device_index = None
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get('maxInputChannels') > 0:
                device_index = i
                self.get_logger().info(f"Using input device {i}: {info.get('name')}")
                break

        if device_index is None:
            raise OSError("No microphone input device found!")

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=FRAMES_PER_BUFFER
        )

        URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"

        try:
            async with websockets.connect(
                URL,
                ping_timeout=20,
                ping_interval=5,
                extra_headers={"Authorization": API_KEY_ASSEMBLY}
            ) as _ws:
                await asyncio.sleep(0.1)
                session_begins = await _ws.recv()
                self.get_logger().info('ASR session started')

                transcripts = []
                stop_event = asyncio.Event()
                last_transcript_time = asyncio.get_event_loop().time()
                silence_timeout = 3
                start_time = asyncio.get_event_loop().time()

                async def send():
                    while not stop_event.is_set():
                        try:
                            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                            data = base64.b64encode(data).decode("utf-8")
                            json_data = json.dumps({"audio_data": data})
                            await _ws.send(json_data)
                        except websockets.exceptions.ConnectionClosedError as e:
                            self.get_logger().error(f'Connection error: {e}')
                            break
                        except Exception as e:
                            self.get_logger().error(f'Send error: {e}')
                            break
                        await asyncio.sleep(0.01)

                async def receive():
                    nonlocal last_transcript_time
                    while not stop_event.is_set():
                        try:
                            result_str = await _ws.recv()
                            result = json.loads(result_str)
                            prompt = result.get("text")
                            if prompt and result.get("message_type") == "FinalTranscript":
                                self.get_logger().info(f'Transcript: {prompt}')
                                transcripts.append(prompt)
                                last_transcript_time = asyncio.get_event_loop().time()
                        except websockets.exceptions.ConnectionClosedError as e:
                            self.get_logger().error(f'Connection error: {e}')
                            break
                        except Exception as e:
                            self.get_logger().error(f'Receive error: {e}')
                            break

                async def check_stop_conditions():
                    while not stop_event.is_set():
                        current_time = asyncio.get_event_loop().time()
                        
                        # Check if duration exceeded
                        if current_time - start_time > duration_seconds:
                            self.get_logger().info('Duration limit reached')
                            stop_event.set()
                            break
                        
                        # Check if silence timeout
                        if current_time - last_transcript_time > silence_timeout:
                            self.get_logger().info('Silence detected')
                            stop_event.set()
                            break
                        
                        await asyncio.sleep(0.5)

                try:
                    send_task = asyncio.create_task(send())
                    receive_task = asyncio.create_task(receive())
                    stop_task = asyncio.create_task(check_stop_conditions())
                    
                    await asyncio.gather(send_task, receive_task, stop_task)
                except asyncio.CancelledError:
                    pass
                finally:
                    stop_event.set()

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        return " ".join(transcripts)


def main():
    rclpy.init()

    asr_service = ASRService()

    rclpy.spin(asr_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()