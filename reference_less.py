import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import padasip as pa

class ReferenceLess:

    # Constructor
    def __init__(self, noise_file_path, controller, num_timesteps=400000):
        self.noise_file_path = noise_file_path
        self.fs, self.reference_noise = wav.read(self.noise_file_path)
        self.n = num_timesteps
        
        # If the noise is 2D, convert it to 1D by choosing left signal always
        if len(self.reference_noise.shape) != 1:
            self.reference_noise = self.reference_noise[:,0]
        # Preprocess reference noise for signal processing by scaling from -1 to 1
        self.reference_noise = self.reference_noise / np.max(np.abs(self.reference_noise))
        self.reference_noise = self.reference_noise[:400000]
        self.n = len(self.reference_noise)

        self.controller = controller

        # Liveness monitor
        self.monitor_window_size = 1000
        self.monitor_liveness_value = 0.70

    # Simulation step
    def simulate(self, output_file_name="output", new_controller=None):
        if new_controller != None:
            self.controller = new_controller
        
        print(f"Simulating for {self.n} timesteps with {self.controller.name}")

        error_mic = list(np.zeros(self.n))
        error_values = np.zeros(self.n)
        liveness_is_satisfied = False
        for i in range(self.n):
            # measure input source wav
            inp = self.reference_noise[i]

            # Send input source to controller
            controller_output = self.controller.input(inp)

            # Error microphone convolves signals from controller and original source
            error_microphone = inp + controller_output
            error_mic[i] = error_microphone

            # Monitor for liveness
            error_values[i] =  abs(error_mic[i] / inp) if inp != 0 else 0
            #error_values = error_values[1:]

            # Only check monitor once liveness window is filled, and only check it once
            if i >= self.monitor_window_size and liveness_is_satisfied == False and np.average(error_values[:self.monitor_window_size]) < self.monitor_liveness_value:
                liveness_is_satisfied = True

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
        if liveness_is_satisfied:
            print(f"Liveness was satisfied for Controller : {self.controller.name}")
        else:
            print(f"Liveness was NOT satisfied for Controller : {self.controller.name}")
        
        # Return difference between error_mic and input signal at each step
        return error_mic


    # Plot reference noise to cancelled noise (ie, error microphone)
    def plot_errors(self, error, output_file_name="reference-less-output"):
        plt.figure(figsize=(12.5,6))
        plt.plot(self.reference_noise, "r", label="original sound")
        plt.plot(error, "b", label="error microphone")
        plt.title(f"ReferenceLess Simulation with {self.controller.name}")
        plt.xlabel("Timestep")
        plt.ylabel("Scaled Amplitude")
        plt.legend()
        plt.savefig(f"{output_file_name}-less.png")
    
    def plot_signals(self, signals, output_file_name="reference-less-output"):
        plt.figure(figsize=(12.5,6))
        for signal in signals:
            plt.plot(signal[0], signal[2], label=signal[1])
        plt.title(f"ReferenceLess Simulation Amplitude Comparison")
        plt.xlabel("Timestep")
        plt.ylabel("Scaled Amplitude")
        plt.legend()
        plt.savefig(f"{output_file_name}.png")
        plt.show()
            
