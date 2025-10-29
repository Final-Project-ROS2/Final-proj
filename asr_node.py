import pyaudio
import websockets
import asyncio
import base64
import json
from api import API_KEY_ASSEMBLY

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Empty
import threading


FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


class ASRPublisher(Node):

    def __init__(self):
        super().__init__('asr_publisher')
        
        # Create publisher for transcripts
        self.transcript_publisher = self.create_publisher(String, 'asr_node', 10)
        
        # Create publisher for emergency stop
        self.emergency_publisher = self.create_publisher(Empty, 'emergency', 10)
        
        # Parameters for emergency stop keywords
        self.declare_parameter('emergency_keywords', ['stop', 'halt', 'emergency'])
        self.emergency_keywords = self.get_parameter('emergency_keywords').value
        
        self.get_logger().info('ASR Publisher Node initialized')
        self.get_logger().info(f'Publishing to topic: voice_command')
        self.get_logger().info(f'Emergency stop topic: emergency')
        self.get_logger().info(f'Emergency keywords: {self.emergency_keywords}')
        
        # Start ASR in separate thread
        self.asr_thread = threading.Thread(target=self.run_asr_thread, daemon=True)
        self.asr_thread.start()

    def run_asr_thread(self):
        """Run ASR in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.continuous_asr())
        except Exception as e:
            self.get_logger().error(f'ASR thread error: {str(e)}')
        finally:
            loop.close()

    async def continuous_asr(self):
        """Continuously run ASR and publish transcripts"""
        p = pyaudio.PyAudio()

        # Find input device
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

        while rclpy.ok():
            try:
                async with websockets.connect(
                    URL,
                    ping_timeout=20,
                    ping_interval=5,
                    extra_headers={"Authorization": API_KEY_ASSEMBLY}
                ) as _ws:
                    await asyncio.sleep(0.1)
                    session_begins = await _ws.recv()
                    self.get_logger().info('ASR session started, listening...')

                    stop_event = asyncio.Event()

                    async def send():
                        while not stop_event.is_set() and rclpy.ok():
                            try:
                                data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                                data = base64.b64encode(data).decode("utf-8")
                                json_data = json.dumps({"audio_data": data})
                                await _ws.send(json_data)
                            except Exception as e:
                                self.get_logger().error(f'Send error: {e}')
                                break
                            await asyncio.sleep(0.01)

                    async def receive():
                        while not stop_event.is_set() and rclpy.ok():
                            try:
                                result_str = await _ws.recv()
                                result = json.loads(result_str)
                                prompt = result.get("text")
                                if prompt and result.get("message_type") == "FinalTranscript":
                                    self.get_logger().info(f'Transcript: {prompt}')
                                    
                                    # Publish immediately
                                    msg = String()
                                    msg.data = prompt
                                    self.transcript_publisher.publish(msg)
                                    self.get_logger().info(f'Published: "{prompt}"')
                            except Exception as e:
                                self.get_logger().error(f'Receive error: {e}')
                                break

                    try:
                        send_task = asyncio.create_task(send())
                        receive_task = asyncio.create_task(receive())
                        
                        await asyncio.gather(send_task, receive_task)
                    except asyncio.CancelledError:
                        pass
                    finally:
                        stop_event.set()

            except Exception as e:
                self.get_logger().error(f'Connection error: {e}')
                self.get_logger().info('Reconnecting in 3 seconds...')
                await asyncio.sleep(3)

        stream.stop_stream()
        stream.close()
        p.terminate()


def main():
    rclpy.init()

    asr_publisher = ASRPublisher()

    try:
        rclpy.spin(asr_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        asr_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()