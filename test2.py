import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtChart import QChart, QChartView, QLineSeries
from pyqtgraph import ImageView

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the title of the window
        self.setWindowTitle("PyQtChart Example")

        # Set the dimensions of the window
        self.setGeometry(100, 100, 800, 600)

        # Create a QChart object
        chart = QChart()

        # Create a QLineSeries object
        series = QLineSeries()

        # Add data to the series
        series.append(0, 6)
        series.append(2, 4)
        series.append(3, 8)
        series.append(7, 4)
        series.append(10, 5)

        # Add the series to the chart
        chart.addSeries(series)

        # Create a QChartView object
        chartView = QChartView(chart)

        # Create an ImageView widget to display the spectrogram image
        spectrogram = np.random.rand(200, 100)
        imageView = ImageView(self)
        imageView.setImage(spectrogram)

        # Create a QVBoxLayout to hold the ImageView and chartView widgets
        widget = QWidget()
        layout = QVBoxLayout()

        # Add the ImageView and chartView widgets to the layout
        layout.addWidget(imageView)
        layout.addWidget(chartView)

        # Set the layout for the widget
        widget.setLayout(layout)

        # Set the widget as the central widget of the window
        self.setCentralWidget(widget)

# Create an instance of QApplication
app = QApplication(sys.argv)

# Create an instance of Window
window = Window()

# Show the window
window.show()

# Run the event loop of QApplication
sys.exit(app.exec_())