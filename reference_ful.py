from distutils.log import error
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import padasip as pa

class ReferenceFul:

    # Constructor
    def __init__(self, noise_file_path, reference_file_path, controller, num_timesteps = 400000):
        self.noise_file_path = noise_file_path
        self.fs, self.input_noise = wav.read(self.noise_file_path)
        self.n = num_timesteps
        
        # If the noise is 2D, convert it to 1D by choosing left signal always
        # if len(self.input_noise.shape) != 1:
        #     self.input_noise = self.input_noise[:,0]
        self.input_noise = self.input_noise.ravel()
        # Preprocess input background noise for signal processing by scaling from -1 to 1
        self.input_noise = self.input_noise / np.max(np.abs(self.input_noise))
        self.input_noise = self.input_noise[:self.n]

        self.reference_file_path = reference_file_path
        _, self.reference_noise = wav.read(self.reference_file_path)
        
        # If the noise is 2D, convert it to 1D by choosing left signal always
        # if len(self.reference_noise.shape) != 1:
        #     self.reference_noise = self.reference_noise[:,0]
        self.reference_noise = self.reference_noise.ravel()
        # Preprocess reference noise for signal processing by scaling from -1 to 1
        self.reference_noise = self.reference_noise / np.max(np.abs(self.reference_noise))
        self.reference_noise = self.reference_noise[:self.n]

        # Combine background noise file with reference file to make "noisy" simulation atmosphere
        # Essentially, combine background chatter with reference music
        self.input_noise = self.input_noise + self.reference_noise

        self.controller = controller

        # Liveness monitor
        self.monitor_window_size = 1000
        self.monitor_liveness_value = .1

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
            error_values =  np.append(error_values, abs(1 - abs(error_mic[i] / self.reference_noise[i])) / 100 if self.reference_noise[i] != 0 else 1)
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
        
        # Set MSE between error_mic and reference signal
        self.mse = (np.square(self.reference_noise - error_mic)).mean()
        print(f"MSE for {self.controller.name} was {self.mse}")


        #self.plot_spectrogram(error_mic, self.fs, output_file_name)
        # FFT
        # self.fft_error_mic = np.fft.fft(error_mic)
        # self.fft_reference_noise = np.fft.fft(self.reference_noise)
        return error_mic

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
        plt.savefig(f"{output_file_name}-ful.png")
    
    def plot_signals(self, signals, output_file_name="reference-ful-output"):
        num_signals = len(signals)

        # Create a subplot grid with 2 rows and 2 columns
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

        for i in range(num_signals):
            row = i // 2  # Calculate the row index
            col = i % 2   # Calculate the column index

            # Calculate the spectrogram for each signal
            f, t, Sxx, im = axs[row, col].specgram(signals[i][0], Fs=self.fs, cmap='viridis', aspect='auto')#, vmin=0, vmax=np.max(np.abs(signals[i])))

            # Set the frequency range to 0-10000 Hz
            axs[row, col].set_ylim(0, 10000)

            # Add colorbar for amplitude
            cbar = fig.colorbar(im, ax=axs[row, col])
            cbar.set_label('Amplitude')

            # Label axes
            axs[row, col].set_ylabel(f'Frequency (Hz)')
            axs[row, col].set_title(signals[i][1])

        # Label common x-axis
        axs[-1, 0].set_xlabel('Time (s)')
        axs[-1, 1].set_xlabel('Time (s)')
        axs[-2, 0].set_xlabel('Time (s)')
        axs[-2, 1].set_xlabel('Time (s)')

        # Save or display the plot
        plt.savefig(f"{output_file_name}_spectrograms.png")
        plt.show()
            
