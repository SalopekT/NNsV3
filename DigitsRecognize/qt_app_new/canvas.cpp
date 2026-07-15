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
    lastPoint = event->pos();
    QPainter painter(&canvasImage);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(Qt::NoPen);
    painter.setBrush(Qt::black);
    int radius=20;
    painter.drawEllipse(lastPoint, radius, radius);
    update();
    drawing = true;
}

void Canvas::mouseMoveEvent(QMouseEvent *event){
    if (!drawing) return;
    QPainter painter(&canvasImage);
    painter.setRenderHint(QPainter::Antialiasing, true);
    int radius=20;
    QPen pen(Qt::black, radius * 2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
    painter.setPen(pen);
    painter.drawLine(lastPoint, event->pos());
    lastPoint = event->pos();
    update();
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

void Canvas::mouseReleaseEvent(QMouseEvent *event){}

std::vector<unsigned char> Canvas::get28x28()
{
    QImage gray = canvasImage.convertToFormat(QImage::Format_Grayscale8);

    int minX = gray.width(), maxX = 0, minY = gray.height(), maxY = 0;
    for (int y = 0; y < gray.height(); ++y) {
        for (int x = 0; x < gray.width(); ++x) {
            int val = qGray(gray.pixel(x, y));
            if (val < 250) {
                minX = std::min(minX, x);
                maxX = std::max(maxX, x);
                minY = std::min(minY, y);
                maxY = std::max(maxY, y);
            }
        }
    }

    std::vector<unsigned char> pixels(28*28, 0);

    if (maxX < minX || maxY < minY) {
        // Nothing drawn — return blank
        return pixels;
    }

    QImage cropped = gray.copy(minX, minY, maxX - minX + 1, maxY - minY + 1);

    int w = cropped.width(), h = cropped.height();
    int newW, newH;
    if (w > h) {
        newW = 20;
        newH = std::max(1, static_cast<int>(std::round(20.0 * h / w)));
    } else {
        newH = 20;
        newW = std::max(1, static_cast<int>(std::round(20.0 * w / h)));
    }
    QImage scaled = cropped.scaled(newW, newH, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

    QImage canvas28(28, 28, QImage::Format_Grayscale8);
    canvas28.fill(Qt::white);
    QPainter p(&canvas28);
    int offsetX = (28 - newW) / 2;
    int offsetY = (28 - newH) / 2;
    p.drawImage(offsetX, offsetY, scaled);
    p.end();

    for (int y = 0; y < 28; ++y)
        for (int x = 0; x < 28; ++x)
            pixels[y*28 + x] = 255 - QColor(canvas28.pixel(x, y)).red();

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
