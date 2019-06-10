#!/usr/bin/env python
#
#  Flask socket io server based on the example from Miguel's excellent Flask-SocketIO.
#  https://github.com/miguelgrinberg/Flask-SocketIO/
# 

from threading import Lock
from flask import Flask, render_template, session, request
from flask_socketio import SocketIO, emit
from wgomoku import GomokuBoard, Heuristics, HeuristicGomokuPolicy, ThreatSearch

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()

@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)


@socketio.on('move', namespace='/test')
def move(message):
    move = {'x': message['x'], 'y': message['y']}
    session['stones'].append(move)
    session['board'] = message['board']
    gomokuBoard = session['gomokuBoard']
    policy = session['adversary']
    gomokuBoard.set(message['x'], message['y'])
    emit('update', {'board': session['board'], 'stones': session['stones']})
    move = policy.suggest(gomokuBoard)
    session['stones'].append({'x': str(move.x), 'y': str(move.y)})
    print(move)
    if move.status == 0:
        gomokuBoard.set(move.x, move.y)
        emit('update', {'board': session['board'], 'stones': session['stones']})


@socketio.on('board_size', namespace='/test')
def set_board_size(message):    
    board = session['board']
    board['size'] = message['size']
    board['squares'] = message['squares']
    emit('update', {'board': session['board'], 'stones': session['stones']})


@socketio.on('connect', namespace='/test')
def test_connect():
    emit('my_response', {'data': 'Connected', 'count': 0})
    session['stones'] = [{'x':10 , 'y': 10},{'x':11 , 'y':11 },{'x': 9, 'y': 11},{'x': 11, 'y': 9}]
    session['board'] = {'size': 600, 'squares': 19}
    h = Heuristics(kappa=3.0)
    session['gomokuBoard'] = GomokuBoard(h, N=19, disp_width=10, stones=[(10, 10),(11, 11),(9, 11),(11, 9)])
    session['adversary'] = HeuristicGomokuPolicy(style = 1, bias=2.0, topn=5, threat_search=ThreatSearch(3,3))
    emit('update', {'board': session['board'], 'stones': session['stones']})


if __name__ == '__main__':
    socketio.run(app, debug=True)
