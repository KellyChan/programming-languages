#!/usr/bin/env python
import pika
import sys

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))

channel = connection.channel()
channel.queue_declare(queue='task_queue', durable=True)

message = ' '.join(sys.argv[1:]) or "New Task!"
channel.basic_publish(exchange='', 
                      routing_key='task_queue', 
                      body=message,
                      properties=pika.BasicProperties(delivery_mode = 2,)   # make message persisitent
                      )

print "[x] sent %r"  % (message,)


connection.close()
