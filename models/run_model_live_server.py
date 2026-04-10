"""
Model script acts as server.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # allows for import of "shared"

import pygame
import threading

import socket
import torch
import random

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
        self.width, self.height = 400, 260
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Live Model Outputs")

        self.font = pygame.font.SysFont("Arial", 14)

        self.wiimote_img = pygame.image.load("assets/wiimote.png")
        self.wiimote_img = pygame.transform.scale(self.wiimote_img, (400, 400))

        self.wiiwheel_img = pygame.image.load("assets/wiiwheel.png")
        self.wiiwheel_img = pygame.transform.scale(self.wiiwheel_img, (120, 120))

        self.telemetry = [0]*8
        self.running = True

        self.buttons = {
            "2": (320, 49),
            "1": (287, 49),
            "PLUS": (192, 25),
            "UP": (55, 49),
            "DOWN": (95, 49),
            "LEFT": (75, 30),
            "RIGHT": (75, 70),
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
        rect = self.wiimote_img.get_rect(center=(self.width//2, 50))
        self.screen.blit(self.wiimote_img, rect)

        for name, pos in self.buttons.items():
            pressed = self.is_pressed(name)

            base_color = (0, 255, 0) if pressed else (155, 165, 155)
            alpha = int(255 * 0.6)  # 20% opacity

            circle_size = 11 if name != "PLUS" else 8

            # Create a temporary surface with alpha
            circle_surface = pygame.Surface((circle_size*2, circle_size*2), pygame.SRCALPHA)
            color_with_alpha = (*base_color, alpha)

            pygame.draw.circle(circle_surface, color_with_alpha, (circle_size, circle_size), circle_size)

            # Blit onto main screen
            self.screen.blit(circle_surface, (pos[0] - circle_size, pos[1] - circle_size))

    def draw_steering(self):
        steer = self.telemetry[7]
        angle = (steer / 14) * 180 - 90

        center_x = 100
        center_y = 175
        
        rotated_image = pygame.transform.rotate(self.wiiwheel_img, -angle)
        rotated_rect = rotated_image.get_rect(center=(center_x, center_y))

        self.screen.blit(rotated_image, rotated_rect.topleft)

    def draw(self):
        self.screen.fill((255, 255, 255))

        # --- BUTTONS ---
        self.draw_buttons()

        # --- STEERING BAR ---
        self.draw_steering()

        # --- RAW TELEMETRY TEXT ---
        for i, val in enumerate(self.telemetry):
            text = self.font.render(f"{i}: {val}", True, (0,0,0))
            self.screen.blit(text, (220, 100 + i*20))

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()

        while self.running:
            events = pygame.event.get()  # grab ALL events once

            for event in events:
                if event.type == pygame.QUIT:
                    self.running = False

            self.draw()

            clock.tick(60)

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