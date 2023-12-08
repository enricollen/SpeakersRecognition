import sounddevice as sd

def list_audio_devices():
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"Device #{i}: {device['name']}")

#list available audio input devices
list_audio_devices()
