from dolphin import memory, event, gui

BASE_ADDRS = {
    "game_base": 0x80000000,
    "player_base": 0x809C18F8,
    "controller_base": 0x809BD70C,
    "position_base": 0x7FFF0000 + 0x9C2EF8
}

FED_INPUTS_FILE_PATH = "../client/input/inputs.csv"
FED_LABELS_FILE_PATH = "../client/input/labels.csv"
CEN_INPUTS_FILE_PATH = "../server/input/inputs.csv"
CEN_LABELS_FILE_PATH = "../server/input/labels.csv"

prev_telemetry = {
    "pos_x": 0,
    "pos_y": 0,
    "pos_z": 0,
    "speed": 0,
    "accel": 0,
    "lap": 1,
}
prev_labels = {
    "1": 0,
    "2": 0,
    "PLUS": 0,
    "DPAD_UP": 0,
    "DPAD_DOWN": 0,
    "DPAD_LEFT": 0,
    "DPAD_RIGHT": 0,
    "STEER": 0    
}

def is_game_loaded():
    """Check if a valid game is running in Dolphin and return its ID."""
    try:
        game_id_str = "".join(
            chr(memory.read_u8(BASE_ADDRS["game_base"] + offset))
            for offset in range(6)
        )
    except Exception as e:
        print(e)
        return False, None

    if len(game_id_str) == 6 and game_id_str.isalnum():
        return True, game_id_str
    else:
        return False, None

def is_in_race():
    """Return True if a race is active."""
    try:
        stage = get_data_point(BASE_ADDRS["player_base"], 0x2B, "u8", deref=True)
        return stage == 1
    except Exception as e:
        print(e)
        return False

def is_game_paused():
    """
    Determines if the game is paused in-game OR pressed home button
    """
    global prev_labels
    return get_data_point(0x809C2F3C, 0x00, "u8", deref=False)

def get_data_point(base_addr, offset, data_type="u8", deref=True):
    """Reads a value from memory, optionally dereferencing a pointer first."""
    try:
        addr = memory.read_u32(base_addr) + offset if deref else base_addr + offset

        if data_type == "u8":
            return memory.read_u8(addr)
        elif data_type == "u16":
            return memory.read_u16(addr)
        elif data_type == "u32":
            return memory.read_u32(addr)
        elif data_type == "f32":
            return memory.read_f32(addr)
        elif data_type == "f64":
            return memory.read_f64(addr)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    except Exception as e:
        print(f"Error reading memory at {hex(addr)}: {e}")
        return None

def get_player_position(pointer_addr=0x809C2EF8):
    """
    Resolves the player position through double pointer dereferencing,
    following the same logic as the C# code.
    """
    try:
        ptr1 = memory.read_u32(pointer_addr)
        if not ptr1:
            return None

        ptr2 = memory.read_u32(ptr1 + 0x40)
        if not ptr2:
            return None

        x = memory.read_f32(ptr2 + 0x0)
        y = memory.read_f32(ptr2 + 0x4)
        z = memory.read_f32(ptr2 + 0x8)

        return {"x": x, "y": y, "z": z}

    except Exception as e:
        print(f"Error reading position: {e}")
        return None

def get_current_race_telemetry(prev_telemetry):
    """Return the current race telemetry from Mario Kart Wii (PAL)."""
    player_pos = get_player_position()

    dx = abs(prev_telemetry["pos_x"] - player_pos["x"])
    dy = abs(prev_telemetry["pos_y"] - player_pos["y"])
    dz = abs(prev_telemetry["pos_z"] - player_pos["z"])

    speed = (dx**2 + dy**2 +dz**2) ** 0.5

    lap_progress = get_data_point(0x809BD730, 0xF8, "f32", deref=True)

    telemetry = {
        "pos_x": player_pos["x"],
        "pos_y": player_pos["y"],
        "pos_z": player_pos["z"],
        "speed": speed,
        "accel": speed - prev_telemetry["speed"],
        "lap":   lap_progress - 1,
    }

    return telemetry

def get_current_labels():
    """Gets the current ctrls from the WiiMote."""
    controller_inputs = get_data_point(BASE_ADDRS["controller_base"], 0x61, "u8", deref=True)
    controller_steer = get_data_point(BASE_ADDRS["controller_base"], 0x3C, "u8", deref=True)

    normalised_labels = { 
        "2":           int((controller_inputs & 1) != 0), 
        "1":           int((controller_inputs & 2) != 0),
        "PLUS":        int((controller_inputs & 4) != 0),
        "DPAD_UP":     int((controller_inputs & 8) != 0),
        "DPAD_DOWN":   int((controller_inputs & 16) != 0),
        "DPAD_LEFT":   int((controller_inputs & 32) != 0),
        "DPAD_RIGHT":  int((controller_inputs & 64) != 0),
        "STEER":       controller_steer / 7
    }

    return normalised_labels

def write_data(data_list, filepath):
    """Writes list of data to the given file."""
    try:
        with open(filepath, "a") as file:
            file.write(",".join(list(map(str, data_list))) + "\n")
    except Exception as e:
        print(e)

def draw_gui(telemetry, labels, frame_count):
    # Telemetry
    environment_inputs = [f"{ctrl}: {state}" for ctrl, state in telemetry.items()]
    gui.draw_text((10, 10), 0xffff0000, "\n".join(environment_inputs))

    # Labels
    controller_inputs = [f"{ctrl}: {state}" for ctrl, state in labels.items()]
    gui.draw_text((10, 95), 0xffff0000, "\n".join(controller_inputs))

    # Frame count
    gui.draw_text((10, 210), 0xffff0000, f"frame: {frame_count}")

async def main():
    """Main function."""
    # Initialise
    global prev_telemetry, prev_labels
    last_game_loaded = False
    last_game_id = None
    last_in_race = False
    last_is_paused = False
    frame_count = 0

    while True:
        # Draw GUI
        draw_gui(prev_telemetry, prev_labels, frame_count)

        # Check game is loaded
        loaded, game_id = is_game_loaded()
        if loaded != last_game_loaded or game_id != last_game_id:
            last_game_loaded = loaded
            last_game_id = game_id

            if loaded:
                print(f"✅ Game loaded: {game_id}")
            else:
                print("❌ Game unloaded.")
                await event.frameadvance()
                continue
        elif not loaded:
            await event.frameadvance()
            continue
        
        # Check in race
        in_race = is_in_race()
        if in_race != last_in_race:
            last_in_race = in_race

            if in_race:
                print("🏁 Entered race.")
            else:
                print("🕹️ Exited race.")
                await event.frameadvance()
                continue
        elif not in_race:
            await event.frameadvance()
            continue

        # Check is paused
        is_paused = is_game_paused()
        if is_paused != last_is_paused:
            last_is_paused = is_paused
            if is_paused:
                print("⏸️ Game paused.")
                await event.frameadvance()
                continue
            else:
                print("▶️ Game unpaused.")
        elif is_paused:
            await event.frameadvance()
            continue

        # Get input data
        telemetry = get_current_race_telemetry(prev_telemetry)
        labels = get_current_labels()

        # Stop recording after race is done
        if telemetry["lap"] > 3.0:
            event.frameadvance()
            continue

        # Overwrite prev
        prev_telemetry = {k: v for k, v in telemetry.items()}
        prev_labels = {k: v for k, v in labels.items()}

        # Update frame count
        frame_count += 1
        telemetry["frame"] = frame_count

        # Federated Data Writing (ONCE THEN DELETED)
        write_data(telemetry.values(), FED_INPUTS_FILE_PATH)
        write_data(labels.values(), FED_LABELS_FILE_PATH)

        # Centralised Data Writing (PERSISTS)
        write_data(telemetry.values(), CEN_INPUTS_FILE_PATH)
        write_data(labels.values(), CEN_LABELS_FILE_PATH)

        await event.frameadvance()


if __name__ == "__main__":
    await main()