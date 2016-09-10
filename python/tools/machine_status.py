#!/usr/bin/env 
# -*- coding: utf-8 -*-

## Project Description:
## return the machine status

__author__ = 'Kelly Chan'
__date__ = 'Spet 29 2014'
__version__ = '1.0.0'


import os
import time
import datetime

import uuid
import socket
import psutil as ps


class MachineStatus(object):

    def __init__(self):
        self.MAC = None
        self.IP = None
        self.cpu = {}
        self.memory = {}
        self.process = {}
        self.network = {}
        self.status = []  # [cpu %, memeroy %, process #, established #]
        self.get_init_info()  # function
        self.get_status_info()  # function
  
  
    #  status
    def run(self):
        self.get_status_info()
        self.save_status_to_db()

    def save_status_to_db(self):
        print self.status

    #  data gathering
    def get_init_info(self):
        self.cpu = {
                        'cores': 0,                # cpu逻辑核数
                        'percent': 0,              # cpu使用率
                        'system_time': 0,          # 内核态系统时间
                        'user_time': 0,            # 用户态时间
                        'idle_time': 0,            # 空闲时间
                        'nice_time': 0,            # nice时间 (花费在调整进程优先级上的时间)
                        'softirq': 0,              # 软件中断时间
                        'irq': 0,                  # 中断时间
                        'iowait': 0                # IO等待时间

                   }

        self.memeory = {
                            'percent': 0,
                            'total': 0,
                            'vailable': 0,
                            'used': 0,
                            'free': 0,
                            'active': 0
                       }


        self.process = {
                            'count': 0,         # 进程数目
                            'pids': 0           # 进程识别号
                       }

        self.network = {
                            'count': 0,         # 连接总数
                            'established': 0    # established连接数
                       }

        self.status = [0,0,0,0]  # [cpu usage %, memeory usage %, process #, established #]

        self.get_mac_address()
        self.get_ip_address()
  

    def get_status_info(self):
        self.get_cpu_info()
        self.get_memory_info()
        self.get_process_info()
        self.get_network_info()
        
        self.status[0] = self.cpu['percent']
        self.status[1] = self.memeory['percent']
        self.status[2] = self.process['count']
        self.status[3] = self.network['established']


    def get_mac_address(self):
        mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
        self.MAC = ":".join([mac[e:e+2] for e in range(0, 11, 2)])

    def get_ip_address(self):
        tempSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tempSock.connect(('8.8.8.8', 80))
        address = tempSock.getsockname()[0]
        tempSock.close()

        self.IP = address


    def get_cpu_info(self):
        self.cpu['cores'] = ps.cpu_count()
        self.cpu['precent'] = ps.cpu_percent(interval=2)
        cpu_times = ps.cpu_times()
        #print cpu_times

        self.cpu['system_time'] = cpu_times.system
        self.cpu['user_time'] = cpu_times.user
        self.cpu['idle_time'] = cpu_times.idle
        #self.cpu['nice_time'] = cpu_times.nice
        #self.cpu['softirq'] = cpu_times.softirq
        #self.cpu['irq'] = cpu_times.irq
        #self.cpu['iowait'] = cpu_times.iowait


    def get_memory_info(self):
        memory_info = ps.virtual_memory()
        #print memory_info
        
        self.memory['percent'] = memory_info.percent
        self.memory['total'] = memory_info.total
        self.memory['vailable'] = memory_info.available
        self.memory['used'] = memory_info.used
        self.memory['free'] = memory_info.free
        #self.memory['active'] = memory_info.active


    def get_process_info(self):
        pids = ps.pids()

        self.process['pids'] = pids 
        self.process['count'] = len(pids)

  
    def get_network_info(self):
        conns = ps.net_connections()
        self.network['count'] = len(conns)

        count = 0
        for conn in conns:
            if conn.status is 'ESTABLISHED':
                count += 1
        self.network['established'] = count

  
if __name__ == '__main__':
    
    MS = MachineStatus()
    print "IP: %s" % MS.IP
    print "MAC: %s" % MS.MAC
    print "CPU: %s" % MS.cpu
    print "Memeory: %s" % MS.memory
    print "Status [cpu usage | memeory usage | process # | established #]: %s" % MS.status

