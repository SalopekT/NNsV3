#include "Canvas.h"
#include <QPainter>
#include <QMouseEvent>
#include <QLabel>

Canvas::Canvas(QWidget *parent)
    : QWidget(parent), drawing(false)
{

    canvasImage = QImage(width(), height(), QImage::Format_RGB32);
    canvasImage.fill(Qt::white);
}

void Canvas::paintEvent(QPaintEvent *event){
    QPainter painter(this);
    painter.drawImage(rect(), canvasImage);
}

void Canvas::mousePressEvent(QMouseEvent *event){
    if (!(event->buttons() & Qt::LeftButton))
        return;

    QPainter painter(&canvasImage);
    painter.setRenderHint(QPainter::Antialiasing, true);

    painter.setPen(Qt::NoPen);
    painter.setBrush(Qt::black);


    int radius = 8;
    painter.drawEllipse(event->pos(), radius, radius);

    update();
    drawing = true;
}
void Canvas::resizeEvent(QResizeEvent *event)
{
    if (canvasImage.size() != event->size()) {
        QImage newImage(event->size(), QImage::Format_RGB32);
        newImage.fill(Qt::white);

        QPainter p(&newImage);
        p.drawImage(QPoint(0,0), canvasImage);

        canvasImage = newImage;
    }

    QWidget::resizeEvent(event);
}
void Canvas::mouseMoveEvent(QMouseEvent *event){
    if (!drawing) return;

    QPainter painter(&canvasImage);
    painter.setRenderHint(QPainter::Antialiasing, true);

    painter.setPen(Qt::NoPen);
    painter.setBrush(Qt::black);

    int radius = 8;
    painter.drawEllipse(event->pos(), radius, radius);

    update();
}
void Canvas::mouseReleaseEvent(QMouseEvent *event){}

std::vector<unsigned char> Canvas::get28x28()
{
    QImage gray = canvasImage.convertToFormat(QImage::Format_Grayscale8);
    QImage small = gray.scaled(28, 28, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

    std::vector<unsigned char> pixels(28*28);
    for (int y = 0; y < 28; ++y)
        for (int x = 0; x < 28; ++x)
            pixels[y*28 + x] = 255 - QColor(small.pixel(x,y)).red();

    return pixels;
}

void Canvas::showTransformation() {
    std::vector<unsigned char> pixels = get28x28();

    // Create the image (ensure it copies the data by using .copy())
    QImage img = QImage(pixels.data(), 28, 28, QImage::Format_Grayscale8).copy();

    // Create a label on the fly
    QLabel* viewer = new QLabel();
    viewer->setWindowTitle("NN Input Preview");
    viewer->setPixmap(QPixmap::fromImage(img).scaled(140, 140, Qt::KeepAspectRatio, Qt::FastTransformation));

    // This makes it a standalone window
    viewer->setAttribute(Qt::WA_DeleteOnClose);
    viewer->show();
}
void Canvas::clearCanvas() {
    // Fill the image with white (all pixels to 255)
    canvasImage.fill(Qt::white);

    // Trigger a repaint to update the screen
    update();
}
