# CS 6376 - Active Noise Cancellation Final Project
Authors: Nathan Hunsberger, Berke Lunstad

This repo is the works completed for an Active Noise Cancellation project completed while at Vanderbilt University. The purpose of this project is to provide a framework for testing/comparing different controllers and their ability to cancel background noise. For background knowledge on the Referenceful vs Referenceless vs Filtered simulation models, please review the `final-paper.pdf` file. This file outlines our approach, and the scope of the simulation tool.

## How to add user-defined controller
To add your own user-defined controller, clone this repo and implement the Controller abstract base class (found in `controllers/` directory). There are two functions inside the Controller class (besides the constructor): `input()` and `feed_forward()`. `input()` is an abstract method, thus all controllers must implement it. `input()` is meant to receive the background noise signal that should be cancelled, and is used by controllers for all simulations. `feed_forward()` is not abstract, and the Controller base class actually defines the method as just a `pass`, essentially a no-op. The `feed_forward()` method is meant to accept the feed forward logic of the Referenceful and Referenceless simulations, thus controllers designed for those simulations should override the method with their feed forward logic (subsequently, Filtered controllers need not override the method). The `feed_forward()` method input will be slightly different depending on the simulation, as the Referenceful simulation will input the reference noise signal, whereas the Referenceless simulation will input the error signal of the previous timestep. The simulation classes will call these functions in order to interface with the user-defined controllers.

## How to run simulations
To run simulations, users can simply run `python3 simulation.py` once cloning the repo (note `requirements.txt` for any library requirements as well). Out-of-the-box, `simulation.py` will run a Referenceful simulation, evaluating the RLS, LMS, and NLMS controllers against `sounds/coffeshop.wav` as the background signal and `sounds/song.wav` as the reference signal. Check the `sound/` directory for more background sounds you can test against. In order to change the simulation type between Referenceful, Referenceless, and Filtered, simply uncomment out the simulation you wish to run (as directed by the comments in `simulation.py`).

If you wish to modify simulation.py to use a custom defined controller that you implement, import the controller to the file, instantiate your controller, and then run the simulation on your controller. For example, if your controller is defined as `custom_controller`, simply execute `custom_controller_difference = simulation.simulate(new_controller=custom_controller)`. This will run whichever simulation you choose on your custom controller. You can add the `custom_controller_difference` variable to the `signals` list at the bottom of `simulation.py` to plot your custom defined controller against the other controllers (RLS, LMS, and NLMS). The Referenceful simulation will generate spectrograms, and the Referenceless/Filtered simulations will generate amplitude graphs (check `results/` for sample plots and wavs). All simulations will generate an `output.wav` file, which you can use to listen to the error signal your controller produced for the given simulation.

## References
Check `final-paper.pdf` for a full list of references. In terms of wav files supplied in the `sounds/` directory, they were retrieved from the below sources:

Sounds downloaded from:

https://freesound.org/people/InspectorJ/sounds/414976/

https://freesound.org/people/InspectorJ/sounds/422051/

https://freesound.org/people/C_Rogers/sounds/453074/

https://pixabay.com/music/search/wav/
