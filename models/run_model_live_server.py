"""
Model script acts as server.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # allows for import of "shared"

import pygame
import math
import threading

import socket
import torch

import shared.preprocessing as pp

MODEL_NAME = "centralised_model"
MODEL_PATH = f"models/{MODEL_NAME}.pt"

HOST = "127.0.0.1"
PORT = 5000


model = pp.generate_base_model()
dicts = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(dicts)


class WiimoteGUI:
    def __init__(self):
        pygame.init()
        self.width, self.height = 600, 1000
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Live Model Outputs")

        self.font = pygame.font.SysFont("Arial", 18)

        # Load Wiimote image (optional)
        try:
            self.wiimote_img = pygame.image.load("wiimote.png")
            self.wiimote_img = pygame.transform.scale(self.wiimote_img, (120, 300))
        except:
            self.wiimote_img = pygame.Surface((120, 300))
            self.wiimote_img.fill((200, 200, 200))

        self.telemetry = [0]*8
        self.running = True

        self.buttons = {
            "2": (200, 450),
            "1": (200, 400),
            "PLUS": (300, 200),
            "UP": (200, 180),
            "DOWN": (200, 320),
            "LEFT": (160, 250),
            "RIGHT": (240, 250),
        }

        self.button_index = {
            "2": 0,
            "1": 1,
            "PLUS": 2,
            "UP": 3,
            "DOWN": 4,
            "LEFT": 5,
            "RIGHT": 6,
        }

    def update_data(self, telemetry):
        if len(telemetry) < 8:
            print("Warning: incomplete telemetry received, ignoring")
            return
        self.telemetry = telemetry

    def is_pressed(self, button_name):
        idx = self.button_index[button_name]
        return int(self.telemetry[idx]) == 1

    def draw_buttons(self):
        for name, pos in self.buttons.items():
            pressed = self.is_pressed(name)

            color = (0, 255, 0) if pressed else (100, 100, 100)
            pygame.draw.circle(self.screen, color, pos, 15)

            label = self.font.render(name, True, (0,0,0))
            rect = label.get_rect(center=pos)
            self.screen.blit(label, rect)

    def draw_steering_bar(self):
        steer = self.telemetry[7]

        bar_x = 50
        bar_y = 550
        bar_width = 300
        bar_height = 20

        pygame.draw.rect(self.screen, (80, 80, 80), (bar_x, bar_y, bar_width, bar_height))

        pos = int((steer / 14) * bar_width)

        pygame.draw.circle(self.screen, (0, 255, 0), (bar_x + pos, bar_y + bar_height//2), 10)

        text = self.font.render(f"STEER: {steer}", True, (255,255,255))
        self.screen.blit(text, (bar_x, bar_y - 25))

    def draw(self):
        self.screen.fill((30, 30, 30))

        # --- STEERING ROTATION ---
        steer = self.telemetry[7]
        normalized = (steer - 7) / 7
        angle = normalized * 60

        rotated = pygame.transform.rotate(self.wiimote_img, angle)
        rect = rotated.get_rect(center=(self.width//2, self.height//2))
        self.screen.blit(rotated, rect)

        # --- BUTTONS ---
        self.draw_buttons()

        # --- STEERING BAR ---
        self.draw_steering_bar()

        # --- RAW TELEMETRY TEXT ---
        for i, val in enumerate(self.telemetry):
            text = self.font.render(f"{i}: {val}", True, (255,255,255))
            self.screen.blit(text, (10, 10 + i*20))

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.draw()
            clock.tick(30)

        pygame.quit()


def socket_loop(gui):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen()

    print(f"Listening on {HOST}:{PORT}")

    conn, addr = server.accept()
    print("Connected:", addr)

    train_min = None
    train_max = None

    while True:
        data = conn.recv(1024)
        if not data:
            break

        parts = data.decode().split()
        telemetry = [float(x) for x in parts[:-1]] + [int(parts[-1])]

        gui.update_data(telemetry)

        telemetry_tensor = torch.tensor(telemetry, dtype=torch.float32).unsqueeze(0)

        if train_min is None:
            train_min = telemetry_tensor.clone()
            train_max = telemetry_tensor.clone()
        else:
            train_min = torch.minimum(train_min, telemetry_tensor)
            train_max = torch.maximum(train_max, telemetry_tensor)

        normalised_telemetry = (telemetry_tensor - train_min) / (train_max - train_min + 1e-8)

        with torch.no_grad():
            outputs = model(normalised_telemetry)

        reply = " ".join(map(str, outputs.tolist()[0]))
        conn.sendall(reply.encode())

    conn.close()

def main():
    gui = WiimoteGUI()

    # Start GUI thread
    gui_thread = threading.Thread(target=gui.run, daemon=True)
    gui_thread.start()

    # Start socket thread
    socket_thread = threading.Thread(target=socket_loop, args=(gui,), daemon=True)
    socket_thread.start()

    # Keep main thread alive
    while gui.running:
        pass

if __name__ == "__main__":
    main()