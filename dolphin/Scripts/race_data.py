from dolphin import memory, event, gui

BASE_ADDRS = {
    "game_base": 0x80000000,
    "controller_base": 0x809BD70C,
}

FED_INPUTS_FILE_PATH = "../client/input/inputs.csv"
FED_LABELS_FILE_PATH = "../client/input/labels.csv"
CEN_INPUTS_FILE_PATH = "../server/input/inputs.csv"
CEN_LABELS_FILE_PATH = "../server/input/labels.csv"

cur_telemetry = {
        "pos_x": 0,
        "pos_y": 0,
        "pos_z": 0,
        
        "speed": 0,
        "accel": 0,

        "lap": 1,
        "frame_count": 0
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
        stage = get_data_point(BASE_ADDRS["plyr_base"], 0x2B, "u8", deref=True)
        return stage == 1
    except Exception as e:
        print(e)
        return False

def is_game_paused(labels):
    """Return True if the race is paused."""
    return labels["PLUS"] == 1

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

def get_current_race_telemetry(prev_telemetry, frame_count):
    """Return the current race telemetry from Mario Kart Wii (PAL)."""
    speed = get_data_point(0x8057AC2C, 0x00, "u8", deref=True)

    telemetry = {
        "pos_x": 0,
        "pos_y": 0,
        "pos_z": 0,
        
        "speed": speed,
        "accel": prev_telemetry["speed"] - speed,

        "lap": 1,
        "frame_count": frame_count
    }

    return telemetry

def get_current_labels():
    """Gets the current ctrls from the WiiMote."""
    controller_inputs = get_data_point(BASE_ADDRS["controller_base"], 0x61, "u8", deref=True)
    controller_steer = get_data_point(BASE_ADDRS["controller_base"], 0x3C, "u8", deref=True)

    normalised_labels = {
        "1":           int((mem_val & 2) != 0),
        "2":           int((mem_val & 1) != 0),
        "PAUSE":       int((mem_val & 4) != 0),
        "DPAD_UP":     int((mem_val & 8) != 0),
        "DPAD_DOWN":   int((mem_val & 16) != 0),
        "DPAD_LEFT":   int((mem_val & 32) != 0),
        "DPAD_RIGHT":  int((mem_val & 64) != 0),
        "steer":       (controller_steer / 7) - 1
    }

    gui.draw_text((10, 30), 0xffff0000, f"moving_dir: {moving_dir}")

    return normalised_labels

def write_data(data_list, filepath):
    """Writes list of data to the given file."""
    try:
        with open(filepath, "a") as file:
            file.write(",".join(list(map(str, data_list))) + "\n")
    except Exception as e:
        print(e)

async def main():
    """Main function."""
    global cur_telemetry
    last_game_loaded = False
    last_game_id = None
    last_in_race = False
    last_is_paused = False
    frame_count = 0

    # Erase data if there
    open(FED_INPUTS_FILE_PATH, "w").close()
    open(FED_LABELS_FILE_PATH, "w").close()

    while True:
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

        in_race = is_in_race()
        if in_race != last_in_race:
            last_in_race = in_race

            if in_race:
                print("🏁 Entered race.")
            else:
                print("🕹️ Exited race.")
                await event.frameadvance()
                continue
        
        if in_race and loaded:
            frame_count += 1
            gui.draw_text((10, 10), 0xffff0000, f"Frame: {frame_count}")

            telemetry = get_current_race_telemetry(cur_telemetry, frame_count)
            labels = get_current_labels()

            is_paused = is_game_paused(labels)
            if is_paused != last_is_paused:
                last_is_paused = is_paused

                if is_paused:
                    print("⏸️ Game paused.")
                else:
                    print("▶️ Game unpaused.")
                    await event.frameadvance()
                    continue
            
            cur_telemetry = telemetry

            # Federated Data Writing (ONCE THEN DELETED)
            write_data(telemetry.values(), FED_INPUTS_FILE_PATH)
            write_data(labels.values(), FED_LABELS_FILE_PATH)

            # Centralised Data Writing (PERSISTS)
            write_data(telemetry.values(), CEN_INPUTS_FILE_PATH)
            write_data(labels.values(), CEN_LABELS_FILE_PATH)

        await event.frameadvance()


if __name__ == "__main__":
    await main()