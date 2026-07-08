#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include "Canvas.h"
#include "../../src/Network.hpp"
#include "../../src/MNISTdata/FileReader.hpp"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
private slots:
    void processDrawing();
    void showNextMNIST();
private:
    Ui::MainWindow *ui;
    Canvas* canvas;
    Network* network;
    QLabel* predictionLabel;
    FileReader* reader;

    QLabel* mnistImageLabel;
    QLabel* mnistInfoLabel;

    int mnistIndex = 0;

    void displayMNISTImage();

};
#endif // MAINWINDOW_H
