import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from tqdm.notebook import tqdm
from IPython.display import Audio
import pyroomacoustics as pra
import numpy as np
import padasip as pa

class ReferenceLess:

    # Constructor
    def __init__(self, noise_file_path, controller):
        self.noise_file_path = noise_file_path
        self.fs, self.reference_noise = wav.read(self.noise_file_path) # Fix reference_noise 2D shape
        self.n = len(self.reference_noise)

        # Preprocess reference noise for signal processing by scaling from -1 to 1
        self.reference_noise = self.reference_noise / np.max(np.abs(self.reference_noise))
        self.controller = controller

    # Simulation step
    def simulate(self, output_file_name="output", new_controller=None):
        if new_controller != None:
            self.controller = new_controller
        
        print(f"Simulating for {self.n} timesteps with {self.controller.name}")

        error = list(np.zeros(self.n))
        for i in tqdm(range(self.n)):
            # measure input source wav
            inp = self.reference_noise[i]

            # Send input source to controller
            controller_output = self.controller.input(inp)

            # Error microphone convolves signals from controller and original source
            error_microphone = inp + controller_output
            error[i] = error_microphone

            # Feed Foward Step -
            # Send error microphone output back to speaker
            # Allows controller to learn if it would like
            self.controller.feed_forward(error_microphone)
        # Write output
        wav.write(f"{output_file_name}.wav", self.fs, np.array(error))
        self.plot_errors(error, output_file_name)
        # ERROR METRIC!

    # Plot reference noise to cancelled noise (ie, error microphone)
    def plot_errors(self, error, output_file_name):
        plt.figure(figsize=(12.5,6))
        plt.plot(self.reference_noise, "r", label="original sound")

        # Normalize error for plotting between -1 and 1
        max_val = max(error)
        error_plt = [v / max_val for v in error]

        plt.plot(error_plt, "b", label="error microphone")
        plt.title(f"ReferenceLess Simulation with {self.controller.name}")
        plt.xlabel("Timestep")
        plt.ylabel("Scaled Amplitude")
        plt.savefig(f"{output_file_name}.png")
            
