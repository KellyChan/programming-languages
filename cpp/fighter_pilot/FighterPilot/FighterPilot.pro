#-------------------------------------------------
#
# QtGame Demo
#
#-------------------------------------------------

TARGET = FighterPilot
TEMPLATE = app

QT       += core gui multimedia

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

SOURCES += main.cpp\
           Game.cpp\
           Player.cpp\
           Bullet.cpp\
           Enemy.cpp\
           Score.cpp\
           Health.cpp

HEADERS  += Game.h\
            Player.h\
            Bullet.h\
            Enemy.h\
            Score.h\
            Health.h

FORMS    += 

#-------------------------------------------------
# Debug and Release
#----------

release: DESTDIR = build/release
debug: DESTDIR = build/debug

OBJECTS_DIR = $$DESTDIR/.obj
MOC_DIR = $$DESTDIR/.moc
RCC_DIR = $$DESTDIR/.qrc
UI_DIR = $$DESTDIR/.ui
