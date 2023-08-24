import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 10  # Duration of recording

my_recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
write('./../data/output.wav', fs, my_recording)  # Save as WAV file