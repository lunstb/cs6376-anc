import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import padasip as pa

class ReferenceFul:

    # Constructor
    def __init__(self, noise_file_path, reference_file_path, controller):
        self.noise_file_path = noise_file_path
        self.fs, self.input_noise = wav.read(self.noise_file_path)
        
        # If the noise is 2D, convert it to 1D by choosing left signal always
        if len(self.input_noise.shape) != 1:
            self.input_noise = self.input_noise[:,0]
        # Preprocess input background noise for signal processing by scaling from -1 to 1
        self.input_noise = self.input_noise / np.max(np.abs(self.input_noise))
        self.input_noise = self.input_noise[:200000]
        self.n = len(self.input_noise)

        self.reference_file_path = reference_file_path
        _, self.reference_noise = wav.read(self.reference_file_path)
        
        # If the noise is 2D, convert it to 1D by choosing left signal always
        if len(self.reference_noise.shape) != 1:
            self.reference_noise = self.reference_noise[:,0]
        # Preprocess reference noise for signal processing by scaling from -1 to 1
        self.reference_noise = self.reference_noise / np.max(np.abs(self.reference_noise))
        self.reference_noise = self.reference_noise[:200000]

        # Combine background noise file with reference file to make "noisy" simulation atmosphere
        # Essentially, combine background chatter with reference music
        self.input_noise = self.input_noise + self.reference_noise

        self.controller = controller

        # Liveness monitor
        self.monitor_window_size = 1000
        self.monitor_liveness_value = .10

    # Simulation step
    def simulate(self, output_file_name="output", new_controller=None):
        if new_controller != None:
            self.controller = new_controller
        
        print(f"Simulating for {self.n} timesteps with {self.controller.name}")

        error_mic = list(np.zeros(self.n))
        error_values = np.zeros(self.monitor_window_size)
        liveness_is_satisfied = False
        for i in range(self.n):
            # measure input source wav
            inp = self.input_noise[i]

            # Send input source to controller
            controller_output = self.controller.input(inp)

            # Error microphone convolves signals from controller and original source
            error_microphone = inp + controller_output
            error_mic[i] = error_microphone

            # Monitor for liveness
            error_values =  np.append(error_values, abs(1 - abs(error_mic[i] / self.reference_noise[i])) if self.reference_noise[i] != 0 else 0)
            error_values = error_values[1:]

            # Only check monitor once liveness window is filled, and only check it once
            if i >= self.monitor_window_size and liveness_is_satisfied == False and np.average(error_values) < self.monitor_liveness_value:
                liveness_is_satisfied = True

            # Feed Foward Step -
            # Send reference signal back to speaker
            # Allows controller to learn if it would like
            # Because this is reference-ful cancelling, send the reference signal
            self.controller.feed_forward(self.reference_noise[i])
        # Write output
        wav.write(f"{output_file_name}.wav", self.fs, np.array(error_mic))

        # Generate plots
        self.plot_errors(error_mic, output_file_name)

        # Report monitor findings
        if liveness_is_satisfied:
            print(f"Liveness was satisfied for Controller : {self.controller.name}")
        else:
            print(f"Liveness was NOT satisfied for Controller : {self.controller.name}")


    # Plot reference noise to cancelled noise (ie, error microphone)
    def plot_errors(self, error, output_file_name):
        plt.figure(figsize=(12.5,6))
        max_val = max(error)
        error_plt = [v / max_val for v in error]
        plt.plot(self.input_noise, "g", label="original noisy signal")
        plt.plot(error_plt, "b", label="error microphone")
        plt.plot(self.reference_noise, "r", label="reference signal")
        plt.title(f"ReferenceLess Simulation with {self.controller.name}")
        plt.xlabel("Timestep")
        plt.ylabel("Scaled Amplitude")
        plt.legend()
        plt.savefig(f"{output_file_name}.png")
            
