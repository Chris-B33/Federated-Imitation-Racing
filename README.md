# Federated-Imitation-Racing

## Description
**Federated-Imitation-Racing** is a project that explores **supervised federated learning** applied to autonomous driving in the context of **Mario Kart Wii**.<br>
This project serves as a proof-of-concept for federated learning. Player actions—such as steering, acceleration, and item usage—are recorded as labeled datasets.<br> 
While the project currently does not use multiple physical clients, it simulates multiple clients by training separate models on local datasets.<br> 
Federated learning techniques then allow these models to collaboratively improve by sharing model weights or gradients, rather than raw gameplay data, preserving privacy while improving performance.

## Features
- **Federated Learning:** Train models across multiple clients without centralizing sensitive gameplay data.
- **Imitation Learning:** Learn to drive by mimicking players.
- **Privacy-Preserving:** Only model are shared; raw data remain local.
- **Simulation-Compatible:** Works with Mario Kart Wii gameplay data recorded from Dolphin emulator or other sources.


## Installation

### Prerequisites
Before running the project, make sure you have the following installed on your system:

1. **Docker Desktop**  
   - Required to run the server and client containers.  
   - Download and install: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

2. **Dolphin Emulator (with Python scripting support)**  
   - Required for recording gameplay and running automated scripts.  
   - Clone: [https://github.com/Felk/dolphin](https://github.com/Felk/dolphin)  
   - This is a branch with Python Scripting support.
   - Place the Dolphin Installation in `dolphin/`.
   - Place any custom Dolphin scripts under `client/dolphin/Scripts/`.

### Training models
##### It is assumed the same map, vehicle and character will be played anytime data is gathered.

1. **Record your game data**
    - Start Dolphin.
    - Load Mario Kart Wii (PAL).
    - Run record_training_data.py through the scripting interface of Dolphin.
    - Play some races.
2. **Run the compose**
    - ```bash
      docker compose up
    - This will train and send a model to the central server.
3. **Repeat at least 3 times**
    - After 3 models are sent, the server will aggregate them together.
    - At the same time, the centralised model will be trained on all that data together at once.
    - Then it will reset its data too to make a fair comparison.
4. **Gather validation data**
    - Run record_training_data.py through the scripting interface of Dolphin.
    - Play some races.
5. **Run comparison**
    - Run compare.py
    - This uses that validation data on both models and compares performance.