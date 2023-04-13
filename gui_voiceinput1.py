
#pip install PyQt5 numpy sounddevice



# from PyQt5.QtCore import QPointF, QSize
# from PyQt5.QtGui import QIcon
# from PyQt5.QtWidgets import QComboBox, QMainWindow, QPushButton, QVBoxLayout, QWidget


import sys
from PyQt5.QtChart import QChart, QChartView, QLineSeries
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QComboBox, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize , pyqtSignal, QObject # this is for interaction between main gu ind testing
import sounddevice as sd
import numpy as np
import threading
from voice_asr import VoiceASR

class FFTRecorderApp(QMainWindow):

    def __init__(self):
        super().__init__()

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


        layout.addStretch(1)

        # Create a label for the plot
        plot_label = QLabel("Data Plot")
        layout.addWidget(plot_label)

        # Create the plot widget and add it to the layout
        self.plot_widget = QChartView()
        layout.addWidget(self.plot_widget)


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
            fft_data = np.fft.rfft(indata[:, 0])
            print(indata.shape)
            #print(fft_data)

        # Open the audio stream and start recording
        with sd.InputStream(samplerate=None, channels=1, blocksize=chunk_size, callback=audio_callback):
            print("Recording started...")
            input("Press Enter to stop recording...")
            print("Recording stopped...")


    def plot_data(self, data):
        # create a QLineSeries object to hold the data
        series = QLineSeries()

        # add data points to the series
        for i in range(len(data)):
            series.append(QPointF(i, data[i]))

        # create a QChart object and add the series to it
        chart = QChart()
        chart.addSeries(series)

        # set the chart title and axis labels
        chart.setTitle("Data Plot")
        chart.setAnimationOptions(QChart.SeriesAnimations)
        chart.createDefaultAxes()
        chart.axes()[0].setTitleText("X Axis")
        chart.axes()[1].setTitleText("Y Axis")

        # create a QChartView object and set the chart as its model
        self.plot_widget.setChart(chart)

        # show the chart view
        self.plot_widget.show()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui_app = FFTRecorderApp()
    gui_app.show()
    sys.exit(app.exec_())


# This code sets up a simple PyQt5 app with a dropdown menu and a record button. When the button is clicked, it starts recording audio in a separate thread, 
# computing the FFT of each chunk, and printing the result. 
# The dropdown menu only prints the selected index for now, which you can implement further later on.