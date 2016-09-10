#include "Game.h"
#include <QTimer>
#include <QGraphicsTextItem>
#include <QFont>

#include "Enemy.h"
#include <QMediaPlayer>
#include <QBrush>
#include <QImage>


Game::Game(QWidget *parent){

    // create the scene
    scene = new QGraphicsScene();
    // make the scene 800*600 instead of infinity by infinity (default))
    scene->setSceneRect(0,0,800,600);
    // setBackgroundBrush(QBrush(QImage(":/images/bg.png")));
  
    // make the newly created scene the scene to visualize
    // (since Game is a QGraphicsView Widget,
    // it can be used to visualize scenes)
    setScene(scene);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setFixedSize(800,600);

}
