
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from pglive import PlotWidget, ImageView


class Gui_TwoSubplots_Pglive(QMainWindow):
    def __init__(self):
        super().__init__()

        # create the plot widgets for the line plot and spectrogram
        self.line_plot = PlotWidget()
        self.spectrogram = ImageView()

        # create the layout for the line plot and spectrogram
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.spectrogram)
        plot_layout.addWidget(self.line_plot)

        # create the central widget and set the main layout
        central_widget = QWidget()
        central_widget.setLayout(plot_layout)
        self.setCentralWidget(central_widget)

        # set a minimum size for the window
        self.setMinimumSize(800, 600)

    def set_line_data(self, x_data, y_data):
        # clear any existing data from the plot
        self.line_plot.clear()

        # create a new plot curve and add it to the plot
        plot_curve = self.line_plot.plot(x_data, y_data)

        # set the title and axis labels
        self.line_plot.setTitle("Line Plot")
        self.line_plot.setLabel("bottom", "X Axis")
        self.line_plot.setLabel("left", "Y Axis")

    def set_spectrogram_data(self, data):
        # set the data in the spectrogram
        self.spectrogram.setImage(data.T)

        # set the title and axis labels
        self.spectrogram.setTitle("Spectrogram")
        self.spectrogram.setLabel("bottom", "Time")
        self.spectrogram.setLabel("left", "Frequency")

    def receive_data(self, image, data):
        # set the image in the spectrogram
        self.spectrogram.setImage(image)

        # set the line plot data
        self.set_line_data(range(len(data)), data)

    def update_data(self, image, data):
        self.receive_data(image, data)