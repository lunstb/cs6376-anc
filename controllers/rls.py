import numpy as np
import padasip as pa
from controllers.controller import Controller

class RLS(Controller):

    # Constructor
    def __init__(self, name):
        super().__init__(name)
        
        # Using Padasip, create an RLS filter
        self.filter_size = 4
        self.rls_filter = pa.filters.FilterRLS(mu=0.9, n=self.filter_size)
        
        # Keep a history window of previous input signals as a state variable 
        self.history_window = np.zeros(self.filter_size)

        # Keep the previous input as a state variable
        self.previous_input = None

    # Given wav_signal, create output signal to cancel input
    def input(self, wav_signal):
        self.previous_input = wav_signal
        self.history_window = np.append(self.history_window, wav_signal)
        self.history_window = self.history_window[1:]

        return -1 * self.rls_filter.predict(self.history_window)

    # Update filter weights based on reference signal and previous input
    def feed_forward(self, reference_signal):
        if self.previous_input != None:
            self.rls_filter.adapt(reference_signal, self.history_window)