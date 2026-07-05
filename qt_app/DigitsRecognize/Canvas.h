#ifndef CANVAS_H
#define CANVAS_H
#include <QWidget>

class Canvas : public QWidget
{
    Q_OBJECT
    private:
        QImage canvasImage;
        QPoint lastPoint;
        bool drawing;
    public:
        explicit Canvas(QWidget *parent = nullptr);
        void paintEvent(QPaintEvent *event) override;
        void mousePressEvent(QMouseEvent *event) override;
        void mouseMoveEvent(QMouseEvent *event) override;
        void mouseReleaseEvent(QMouseEvent *event) override;
        void resizeEvent(QResizeEvent *event) override;

        QImage getCanvasImage() const { return canvasImage; }
        std::vector<unsigned char> get28x28();
        void showTransformation();
        void clearCanvas();
};
#endif // CANVAS_H
