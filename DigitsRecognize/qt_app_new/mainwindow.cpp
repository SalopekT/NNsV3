#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "Canvas.h"
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <Eigen/Dense>
#include "../../src/Network.hpp"
#include "../../src/MNISTdata/FileReader.hpp"


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    network = new Network("C:\\Users\\tinsa\\Projects\\NNsV3\\build\\Debug\\weights7.txt");
    reader = new FileReader(
        "C:\\Users\\tinsa\\Projects\\NNs\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte",
        "C:\\Users\\tinsa\\Projects\\NNs\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte"
        );

    reader->read_mnist();
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

    mnistImageLabel = new QLabel(this);
    mnistImageLabel->setFixedSize(140,140);

    mnistInfoLabel = new QLabel("MNIST: -", this);

    QPushButton* nextButton = new QPushButton("Next MNIST", central);

    layout->addWidget(mnistImageLabel);
    layout->addWidget(mnistInfoLabel);
    layout->addWidget(nextButton);

    connect(nextButton,
            &QPushButton::clicked,
            this,
            &MainWindow::showNextMNIST);

    displayMNISTImage();
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

void MainWindow::displayMNISTImage()
{
    auto& images = reader->get_images();
    auto& labels = reader->get_labels();

    const auto& image = images[mnistIndex];

    Eigen::VectorXd input(image.size());

    for (size_t i = 0; i < image.size(); ++i)
    {
        input(i) = image[i] / 127.5 - 1.0;
    }

    Eigen::VectorXd output = network->forwardPass(input);

    QImage img(28,28,QImage::Format_Grayscale8);
    for (int y = 0; y < 28; ++y)
    {
        for (int x = 0; x < 28; ++x)
        {
            uint8_t pixel = image[y * 28 + x];
            img.setPixel(x, y, qRgb(pixel, pixel, pixel));
        }
    }
    mnistImageLabel->setPixmap(
        QPixmap::fromImage(img)
            .scaled(140,140,Qt::KeepAspectRatio,
                    Qt::FastTransformation)
        );


    Eigen::Index prediction;
    output.maxCoeff(&prediction);


    mnistInfoLabel->setText(
        QString("True: %1  Prediction: %2")
            .arg(labels[mnistIndex])
            .arg(prediction)
        );
}
void MainWindow::showNextMNIST()
{
    mnistIndex++;

    if (mnistIndex >= reader->get_images().size())
        mnistIndex = 0;

    displayMNISTImage();
}

MainWindow::~MainWindow()
{
    delete network;
    delete reader;
    delete ui;
}
