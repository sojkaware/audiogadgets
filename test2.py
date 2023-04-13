import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtChart import QChart, QChartView, QLineSeries
# pip install PyQtChart

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

        # Set the chart view as the central widget of the window
        self.setCentralWidget(chartView)

# Create an instance of QApplication
app = QApplication(sys.argv)

# Create an instance of Window
window = Window()

# Show the window
window.show()

# Run the event loop of QApplication
sys.exit(app.exec_())