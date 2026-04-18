"""
Model script acts as server.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # allows for import of "shared"

import pygame
import threading

import socket
import time
import torch

import shared.preprocessing as pp

MODEL_NAME = "centralised_model"
MODEL_PATH = f"models/{MODEL_NAME}.pt"

HOST = "127.0.0.1"
PORT = 5000


model = pp.generate_base_model()
_bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
if "state_dict" in _bundle:
    model.load_state_dict(_bundle["state_dict"])
    _ns = _bundle["norm_stats"]
    input_mean = torch.tensor(_ns["input_mean"], dtype=torch.float32)
    input_std  = torch.tensor(_ns["input_std"],  dtype=torch.float32)
    tilt_mean  = _ns["tilt_mean"]
    tilt_std   = _ns["tilt_std"]
    has_norm_stats = True
else:
    model.load_state_dict(_bundle)
    has_norm_stats = False
    print("Warning: model has no embedded norm stats — normalization will be inaccurate")


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

        self.telemetry = [0] * 6
        self.buttons = [0] * 7
        self.steer = 7

        self.running = True

        self.button_postions = {
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
        if len(telemetry) != 6:
            print("Error. Incorrect telemetry.")
            return
        self.telemetry = telemetry

    def update_buttons(self, button_states):
        if len(button_states) != 7:
            print("Error. Incorrect button states.")
            return
        self.buttons = button_states

    def update_steer(self, steer_value):
        self.steer = steer_value

    def is_pressed(self, button_name):
        idx = self.button_index[button_name]
        return int(self.buttons[idx]) == 1

    def draw_buttons(self):
        rect = self.wiimote_img.get_rect(center=(self.width//2, 50))
        self.screen.blit(self.wiimote_img, rect)

        for name, pos in self.button_postions.items():
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
        steer = self.steer
        angle = (steer / 14) * 180 - 90

        center_x = 100
        center_y = 185
        
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
            self.screen.blit(text, (220, 115 + i*20))

        # --- STEER VALUE ---
        steer_text = self.font.render(f"Steer: {int(round(self.steer))}", True, (0, 0, 0))
        self.screen.blit(steer_text, (10, self.height - 25))

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
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("Connected:", addr)

    conn_file = conn.makefile('r')

    while True:
        line = conn_file.readline()
        if not line:
            break

        parts = line.strip().split()
        telemetry = [float(x) for x in parts]

        gui.update_data(telemetry)

        telemetry_tensor = torch.tensor(telemetry, dtype=torch.float32).unsqueeze(0)

        normalised_telemetry = (telemetry_tensor - input_mean) / (input_std + 1e-8)

        with torch.no_grad():
            outputs = model(normalised_telemetry)

        binary = (torch.sigmoid(outputs[:, :7]) > 0.5).float()
        gui.update_buttons(binary.tolist()[0])

        steer = outputs[:, 7:] * tilt_std + tilt_mean
        gui.update_steer(steer.item())

        final = torch.cat([binary, steer], dim=1)
        reply = " ".join(map(str, final.tolist()[0])) + "\n"
        conn.sendall(reply.encode())

    conn.close()

def main():
    gui = WiimoteGUI()

    # Start socket thread in background
    socket_thread = threading.Thread(target=socket_loop, args=(gui,), daemon=True)
    socket_thread.start()

    # Run GUI on main thread (required by pygame on Windows)
    gui.run()

if __name__ == "__main__":
    main()