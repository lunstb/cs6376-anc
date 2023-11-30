from reference_less import ReferenceLess
from controllers.nlms import NLMS

nlms = NLMS("NLMS Controller")

simulation = ReferenceLess("sounds/coffeeshop.wav", nlms)

simulation.simulate()