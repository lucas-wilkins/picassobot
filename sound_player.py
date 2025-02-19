import numpy as np
import sounddevice as sd
from scipy.io import wavfile

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from coms import SerialWrapper

# Initialize the current index
current_index = 0

s = SerialWrapper()

def play_audio(filename):

    # Load the .wav file
    sample_rate, data = wavfile.read(filename)

    # Convert to mono if the audio is stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1).astype(data.dtype)

    # Duration of each chunk (100ms)
    chunk_duration = 0.1
    chunk_size = int(sample_rate * chunk_duration)


    # Callback function to play and calculate mean amplitude
    def audio_callback(outdata, frames, time, status):
        global current_index
        if status:
            print(f"Playback status: {status}")

        # Calculate the end index of the current chunk
        end_index = current_index + frames

        # Ensure we don't exceed the data length
        if end_index > len(data):
            end_index = len(data)
            chunk = data[current_index:end_index]
            outdata[:len(chunk)] = chunk.reshape(-1, 1)
            outdata[len(chunk):] = 0  # Zero-fill the remaining part
        else:
            chunk = data[current_index:end_index]
            outdata[:] = chunk.reshape(-1, 1)

        # Calculate and print the mean amplitude
        mean_amplitude = np.mean(np.abs(chunk))

        # Scale to be nice for driving a servo...

        # Truncate of bottom
        if mean_amplitude > 50:
            mean_amplitude -= 50
        else:
            mean_amplitude = 0

        n = mean_amplitude / 1000

        # Cut off top
        if n > 1:
            n = 1


        n *= 90 # Angle range in degs
        n = int(n)

        # print("*"*n)
        s(n)


        # Update the current index for the next chunk
        current_index = end_index


    # Play the audio with the callback function
    with sd.OutputStream(callback=audio_callback, samplerate=sample_rate, channels=1, dtype=data.dtype):
        sd.sleep(int((len(data) / sample_rate) * 1000))  # Sleep for the duration of the audio



if __name__ == "__main__":
    play_audio("audio/test.wav")
    print("Done")