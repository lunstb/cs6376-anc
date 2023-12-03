import numpy as np
import padasip as pa
from controllers.controller import Controller

class LMS(Controller):

    # Constructor
    def __init__(self, name):
        super().__init__(name)
        
        # Using Padasip, create an LMS filter
        self.filter_size = 5
        self.lms_filter = pa.filters.AdaptiveFilter(model="LMS", n=self.filter_size, mu=0.1, w="random")
        
        # Keep a history window of previous input signals as a state variable 
        self.history_window = np.zeros(self.filter_size)

        # Keep the previous input as a state variable
        self.previous_input = None

    # Given wav_signal, create output signal to cancel input
    def input(self, wav_signal):
        self.previous_input = wav_signal
        self.history_window = np.append(self.history_window, wav_signal)
        self.history_window = self.history_window[1:]

        return -1 * self.lms_filter.predict(self.history_window)

    # Update filter weights based on reference signal and previous input
    def feed_forward(self, reference_signal):
        if self.previous_input != None:
            self.lms_filter.adapt(reference_signal, self.previous_input)