import numpy as np
import gui_twosubplots_pglive

# create an instance of the GuiTwoSubplots
gui = Gui_TwoSubplots_Pglive()

# create some sample data
x_data = np.linspace(0, 10, 100)
y_data = np.sin(x_data)
spectrogram_data = np.random.rand(100, 100)

# update the GUI with the data
gui.update_data(spectrogram_data, y_data)