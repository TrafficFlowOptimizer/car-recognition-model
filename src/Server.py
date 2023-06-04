import json
import os
import socketserver
import sys
from pprint import pprint
from uuid import uuid4

# from Model import Model

HOST, PORT = "localhost", 9092

class SingleTCPHandler(socketserver.BaseRequestHandler):
    """One instance per connection. Override handle(self) to customize action."""

    def handle(self):
        # TODO: przyjmować id nagrania i robić request do bazy przez api czy nagranie w bajtach
        load = json.loads(self.request.recv(4016).decode('utf-8'))
        pprint(load)
        
        # model = Model(load)
        # model.run()

        self.request.send(bytes(json.dumps(data), 'UTF-8'))
        self.request.close()


class SimpleServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = allow_reuse_address = True
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass):
        socketserver.TCPServer.__init__(self, server_address, RequestHandlerClass)
        self.model = None


server = SimpleServer((HOST, PORT), SingleTCPHandler)

try:
    server.serve_forever()
except KeyboardInterrupt:
    sys.exit(0)