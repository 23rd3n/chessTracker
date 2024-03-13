import tkinter as tk
import threading
import sys
import json
import datetime

# Chess board dimensions
DIMENSION = 8
SQ_SIZE = 64
WIDTH = HEIGHT = DIMENSION * SQ_SIZE
PADDING = 200  # Space for player times and custom text

# Window setup
root = tk.Tk()
canvas = tk.Canvas(root, width=WIDTH + 2 * PADDING, height=HEIGHT)
canvas.pack()

def draw_board(canvas):
    colors = ["white", "gray"]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[(r + c) % 2]
            canvas.create_rectangle(PADDING + c * SQ_SIZE, r * SQ_SIZE,
                                    PADDING + (c + 1) * SQ_SIZE, (r + 1) * SQ_SIZE, fill=color)

def draw_player_info(canvas, p1_time, p2_time):
    global p1_time_text_id, p2_time_text_id
    if p1_time_text_id:
        canvas.delete(p1_time_text_id)
    if p2_time_text_id:
        canvas.delete(p2_time_text_id)
    p1_time_text_id = canvas.create_text(70, 40, text=f"Player 1\nTime: {p1_time}", anchor="center", font=("Arial", 16))
    p2_time_text_id = canvas.create_text(WIDTH + PADDING + 80, 40, text=f"Player 2\nTime: {p2_time}", anchor="center", font=("Arial", 16))


def draw_pieces(canvas, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != "--":
                img_path = "/home/student/pphauRos1_ws/images/" + piece + ".png"
                piece_image = tk.PhotoImage(file=img_path)
                canvas.create_image(PADDING + c * SQ_SIZE, r * SQ_SIZE, anchor='nw', image=piece_image)
                images.append(piece_image)

def update_board(canvas, board, p1_time, p2_time):
    canvas.delete("all")
    draw_board(canvas)
    draw_player_info(canvas, p1_time, p2_time)
    draw_pieces(canvas, board)
    root.update()

def handle_incoming_data():
    global active_player
    prev_id = -1
    runOnce = True
    while True:
        input_data = sys.stdin.readline()
        if not input_data:
            break
        data = json.loads(input_data)
        if data['id'] != prev_id:
            handState = data['handState']
            if (handState == 'P1s Hand on the board' or handState == 'P1s Move'):
                active_player = 1
            else: 
                active_player = 2
            # Update only the board and texts, not the time
            root.title(handState)
            if runOnce:
                update_time()
                runOnce = False
            root.after(0, update_board, canvas, data['state_board'], p1_time, p2_time)
            prev_id = data['id']

def update_time():
    global p1_time, p2_time
    if active_player == 1:
        p1_time = update_player_time(p1_time)
    elif active_player == 2:
        p2_time = update_player_time(p2_time)
    # Redraw the player info with the updated time
    draw_player_info(canvas, p1_time, p2_time)
    # Schedule this function to be called again after 1 second (1000 milliseconds)
    root.after(1000, update_time)

def update_player_time(player_time):
    # Assuming player_time is a string in format "HH:MM:SS"
    time_obj = datetime.datetime.strptime(player_time, "%H:%M:%S")
    # Decrement time by one second
    time_obj -= datetime.timedelta(seconds=1)
    return time_obj.strftime("%H:%M:%S")

draw_board(canvas)
images = []

# Initialize times and texts
p1_time = "00:10:00"  
p2_time = "00:10:00"


# Initialize time text IDs
p1_time_text_id = None
p2_time_text_id = None
active_player = 1


# Start the thread for handling incoming data
threading.Thread(target=handle_incoming_data).start()

root.title("Chess")
root.mainloop()

