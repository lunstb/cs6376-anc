import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
#from tqdm.notebook import tqdm
import padasip as pa

class ReferenceLess:

    # Constructor
    def __init__(self, noise_file_path, controller):
        self.noise_file_path = noise_file_path
        self.fs, self.reference_noise = wav.read(self.noise_file_path) # Fix reference_noise 2D shape
        
        # If the noise is 2D, convert it to 1D by choosing left signal always
        if self.reference_noise.shape[1] != 1:
            self.reference_noise = self.reference_noise[:,0]
        # Preprocess reference noise for signal processing by scaling from -1 to 1
        self.reference_noise = self.reference_noise / np.max(np.abs(self.reference_noise))
        self.controller = controller
        self.reference_noise = self.reference_noise[:200000]
        self.n = len(self.reference_noise)

        # Liveness monitor
        self.monitor_window_size = 10000
        self.monitor_liveness_value = 0.50
        self.liveness_is_satisfied = False

    # Simulation step
    def simulate(self, output_file_name="output", new_controller=None):
        if new_controller != None:
            self.controller = new_controller
        
        print(f"Simulating for {self.n} timesteps with {self.controller.name}")

        error_mic = list(np.zeros(self.n))
        error_values = np.zeros(self.monitor_window_size)
        for index, i in enumerate(range(self.n)):
            # measure input source wav
            inp = self.reference_noise[i]

            # Send input source to controller
            controller_output = self.controller.input(inp)

            # Error microphone convolves signals from controller and original source
            error_microphone = inp + controller_output
            error_mic[i] = error_microphone

            # Monitor for liveness
            error_values =  np.append(error_values, abs(error_mic[i] / inp) if inp != 0 else 0)
            error_values = error_values[1:]

            # Only check monitor once liveness window is filled, and only check it once
            if index >= self.monitor_window_size and self.liveness_is_satisfied == False and np.average(error_values) < self.monitor_liveness_value:
                self.liveness_is_satisfied = True

            # Feed Foward Step -
            # Send error microphone output back to speaker
            # Allows controller to learn if it would like
            # Because this is reference-less cancelling, just send the error microphone feedback
            self.controller.feed_forward(error_microphone)
        # Write output
        wav.write(f"{output_file_name}.wav", self.fs, np.array(error_mic))

        # Generate plots
        self.plot_errors(error_mic, output_file_name)

        # Report monitor findings
        if self.liveness_is_satisfied:
            print(f"Liveness was satisfied for Controller : {self.controller.name}")
        else:
            print(f"Liveness was NOT satisfied for Controller : {self.controller.name}")


    # Plot reference noise to cancelled noise (ie, error microphone)
    def plot_errors(self, error, output_file_name):
        plt.figure(figsize=(12.5,6))
        plt.plot(self.reference_noise, "r", label="original sound")
        plt.plot(error, "b", label="error microphone")
        plt.title(f"ReferenceLess Simulation with {self.controller.name}")
        plt.xlabel("Timestep")
        plt.ylabel("Scaled Amplitude")
        plt.legend()
        plt.savefig(f"{output_file_name}.png")
            
