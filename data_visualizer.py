import sys
import numpy as np
from PyQt5.QtChart import QChart, QChartView, QLineSeries
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPainter, QPalette, QKeySequence
from PyQt5.QtWidgets import (QApplication, QGridLayout, QLabel, QMainWindow,
                             QVBoxLayout, QWidget)

from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize


class DataVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Data Visualizer')
        self.setGeometry(200, 100, 800, 600)

        layout = QVBoxLayout()
        grid = QGridLayout()

        self.plot_matrix = self.create_plot_matrix()
        self.plot_line1 = self.create_plot_line()
        self.plot_line2 = self.create_plot_line()
        self.plot_text = self.create_plot_text()

        grid.addWidget(self.plot_matrix, 0, 0, 1, 2)
        grid.addWidget(self.plot_line1, 1, 0)
        grid.addWidget(self.plot_line2, 1, 1)
        grid.addWidget(self.plot_text, 2, 0, 1, 2)

        layout.addLayout(grid)

        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def create_plot_matrix(self):
        label = QLabel(self)
        label.setScaledContents(True)
        return label

    def create_plot_line(self):
        chart = QChart()
        chart_view = QChartView(chart, self)
        chart_view.setRenderHint(QPainter.Antialiasing)
        return chart_view

    def create_plot_text(self):
        label = QLabel(self)
        label.setWordWrap(True)
        return label

    @pyqtSlot(np.ndarray, np.ndarray, float, str)
    def update_data(self, matrix, vector, sinus_sample, text):
        self.update_matrix(matrix)
        self.update_line(vector, self.plot_line1)
        self.update_line([sinus_sample], self.plot_line2)
        self.update_text(text)

    def update_matrix(self, matrix):
        cmap = get_cmap('jet')
        norm = Normalize(vmin=0.0, vmax=1.0)
        img = cmap(norm(matrix))

        qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGBA8888)
       # pixmap = qimage.scaled(self.plot_matrix.size(), Qt.KeepAspectRatio)
      #  self.plot_matrix.setPixmap(pixmap)

    def update_line(self, data, plot):
        series = QLineSeries()
        for i, value in enumerate(data):
            series.append(i, value)

        chart = plot.chart()
        chart.removeAllSeries()
        chart.addSeries(series)
        chart.createDefaultAxes()

    def update_text(self, text):
        self.plot_text.setText(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    data_viz = DataVisualizer()
    data_viz.show()
    sys.exit(app.exec_())