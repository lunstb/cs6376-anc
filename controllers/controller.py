from abc import ABC, abstractmethod

# Base Controller Class -
#   - Abstract Class that all Controllers must implement (Python does not have Abstract classes, so we use the abc library)
# 
# Implements the Controller component that accepts two inputs:
#   - Input 1:
#       - Wav Data:
#           * Discretized wav of input signal
#   - Input 2:
#       - Feed Forward Reference Wav Data:
#           * Discretized wav of reference signal, used if the controller implements feed-forward logic
#
# Produces one output:
#   - Output 1:
#       - Output Wav Data:
#           * Discretized wav of signal the controller will play to cancel the input wav signal
class Controller(ABC):

    # Constructor
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def input(self, wav_signal):
        pass

    def feed_forward(self, reference_signal):
        pass
