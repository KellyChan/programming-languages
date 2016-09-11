# -*- encoding: UTF-8 -*-

import argparse
from naoqi import ALProxy


def main(robot_ip, robot_port, topf_path):
    dialog_p = ALProxy('ALDialog', robot_ip, robot_port)
    dialog_p.setLanguage("English")

    # Load topic - absolute path is required
    topf_path = topf_path.decode('utf-8')
    topic = dialog_p.loadTopic(topf_path.encode('utf-8'))

    # Start dialog
    dialog_p.subscribe('myModule')

    # Activate dialog
    dialog_p.activateTopic(topic)

    raw_input(u"Press 'Enter' to exit.")

    # Deactivate topic
    dialog_p.deactivateTopic(topic)

    # Unload topic
    dialog_p.unloadTopic(topic)

    # Stop dialog
    dialog_p.unsubscribe('myModule')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path",
                        type=str,
                        help="Absolute path of the dialog topic file"
                             " (on the robot).")
    parser.add_argument("--ip",
                        type=str,
                        default="127.0.0.1",
                        help="Robot ip address.")
    parser.add_argument("--port",
                        type=int,
                        default=9559,
                        help="Robot port number.")

    args = parser.parse_args()
    main(args.ip, args.port, args.path)
