#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "Canvas.h"
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <Eigen/Dense>
#include "../../src/Network.hpp"


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    network = new Network("C:\\Users\\tinsa\\Projects\\NNsV3\\build\\Debug\\weights5.txt");
    QWidget *central = new QWidget(this);
    setCentralWidget(central);

    canvas = new Canvas(central);
    QPushButton *processButton = new QPushButton("Process Drawing", central);
    QPushButton *clearButton = new QPushButton("Clear Screen", this);
    predictionLabel = new QLabel("Prediction: -", this);
    QVBoxLayout *layout = new QVBoxLayout(central);
    layout->addWidget(canvas,1);
    layout->addWidget(processButton,0);
    layout->addWidget(clearButton,0);
    layout->addWidget(predictionLabel,0);

    layout->setContentsMargins(0, 0, 0, 0);
    connect(processButton, &QPushButton::clicked, this, &MainWindow::processDrawing);
    connect(clearButton, &QPushButton::clicked, canvas, &Canvas::clearCanvas);
}

void MainWindow::processDrawing()
{
    std::vector<unsigned char> pixels = canvas->get28x28();
    canvas->showTransformation();
    Eigen::VectorXd vec(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
        vec(i) = static_cast<double>(pixels[i]) / 127.5 - 1;
    }


    Eigen::VectorXd output = network->forwardPass(vec);

    int predicted = 0;
    double maxVal = output(0);
    for (int i = 1; i < output.size(); ++i) {
        if (output(i) > maxVal) {
            maxVal = output(i);
            predicted = i;
        }
    }

    this->predictionLabel->setText(QString("Prediction: %1").arg(predicted));
    this->predictionLabel->setStyleSheet("font-size: 24pt; font-weight: bold; color: white;");
}

MainWindow::~MainWindow()
{
    delete ui;
}
