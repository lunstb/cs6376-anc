import numpy as np
import padasip as pa
from controller import Controller

class NLMS(Controller):

    # Constructor
    def __init__(self, name):
        super().__init__(name)
        
        # Using Padasip, create an NLMS filter
        self.filter_size = 10
        self.nlms_filter = pa.filters.AdaptiveFilter(model="NLMS", n=self.filter_size, mu=0.05, w="random")
        
        # Keep a history window of previous input signals as a state variable 
        self.history_window = np.zeros(self.filter_size)

        # Keep the previous input as a state variable
        self.previous_input = None

    # Given wav_signal, create output signal to cancel input
    def input(self, wav_signal):
        self.previous_input = wav_signal
        self.history_window = np.append(self.history_window, wav_signal)
        self.history_window = self.history_window[1:]

        return self.nlms_filter.predict(self.history_window)

    # Update filter weights based on reference signal and previous input
    def feed_forward(self, reference_signal):
        if self.previous_input != None:
            self.nlms_filter.adapt(reference_signal, self.previous_input)