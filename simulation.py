from reference_less import ReferenceLess
from reference_ful import ReferenceFul
import matplotlib.pyplot as plt
from controllers.nlms import NLMS
from controllers.rls import RLS
from controllers.lms import LMS

nlms = NLMS("NLMS Controller")
lms = LMS("LMS Controller")
rls = RLS("RLS Controller")

# Uncomment out the simulation you wish to run!

# Run a ReferenceFul Simulation
simulation = ReferenceFul("sounds/coffeeshop.wav", "sounds/song.wav", rls)

# Run a ReferenceLess simulation
#simulation = ReferenceLess("sounds/coffeeshop.wav", rls)

rls_difference = simulation.simulate()
nlms_difference = simulation.simulate(new_controller=nlms)
lms_difference = simulation.simulate(new_controller=lms)

# Create signals for plotting
signals = [(simulation.reference_noise, "Reference Noise", "y"), (rls_difference, "RLS Controller", "b"), (nlms_difference, "NLMS Controller", "g"), (lms_difference, "LMS Controller", "r")]
simulation.plot_signals(signals)