
#pip install PyQt5 numpy sounddevice

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QComboBox, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize , pyqtSignal, QObject # this is for interaction between main gu ind testing
import sounddevice as sd
import numpy as np
import threading


class VoiceInputApp(QMainWindow):
    data_signal = pyqtSignal(np.ndarray)


    def __init__(self):
        super().__init__()
        # super(First_Window, self).__init__()

        # Signal for sending data
        
        
        # Set up the user interface
        self.init_ui()

    def init_ui(self):
        # Set fixed window size
        self.setFixedSize(300, 400)

        # Set main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # Create the dropdown menu
        self.dropdown_menu = QComboBox()
        layout.addWidget(self.dropdown_menu)
        self.dropdown_menu.addItem("Menu Item 1")
        self.dropdown_menu.addItem("Menu Item 2")
        self.dropdown_menu.activated.connect(self.menu_callback)

        # Create the record button with an image
        self.record_button = QPushButton()
        self.record_button.setIcon(QIcon('icon_mic.png'))
        self.record_button.setIconSize(QSize(150, 150))
        self.record_button.clicked.connect(self.record_audio)
        layout.addWidget(self.record_button)

        # Set the layout
        main_widget.setLayout(layout)

    def menu_callback(self, index):
        # Placeholder function for dropdown menu callback
        print(f"Menu item {index + 1} selected")

    def record_audio(self):
        # Start a new thread for recording audio and computing FFT
        threading.Thread(target=self.record_and_compute_fft).start()

    def record_and_compute_fft(self):
        # Set the chunk size and create a callback for the audio stream
        chunk_size = 4096

        def audio_callback(indata, frames, time, status):
            # Compute the FFT of the audio chunk
            fft_data = np.absolute(np.fft.rfft(indata[:, 0]))
            #print(indata.shape)
            #print(fft_data) 
            #print("Emited")
            self.data_signal.emit(indata[:, 0])


        # Open the audio stream and start recording
        with sd.InputStream(samplerate=None, channels=1, blocksize=chunk_size, callback=audio_callback):
            print("Recording started...")
            input("Press Enter to stop recording...")
            print("Recording stopped...")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui_app = VoiceInputApp()
    gui_app.show()

    from test2 import GuiTwoSubplotsImageLinePlot
    debug_gui = GuiTwoSubplotsImageLinePlot()
    debug_gui.show()

    # Connect the VoiceInputApp's fft_data_signal to the OtherGUI's update_plot method
    gui_app.data_signal.connect(debug_gui.update_plot)
    #gui_app.data_signal.connect(debug_gui.data_signal.emit)
    sys.exit(app.exec_())


# This code sets up a simple PyQt5 app with a dropdown menu and a record button. When the button is clicked, it starts recording audio in a separate thread, 
# computing the FFT of each chunk, and printing the result. 
# The dropdown menu only prints the selected index for now, which you can implement further later on.