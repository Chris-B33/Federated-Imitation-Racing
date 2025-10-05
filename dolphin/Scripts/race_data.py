from dolphin import memory, event, gui

BASE_ADDRS = {
    "game_base": 0x80000000,
    "plyr_base": 0x809C18F8,
    "dir_base": 0x809BD70C,
    "lap_base": 0x809BD730,
    "pos_base": 0x809C2EF8
}

INPUTS_FILE_PATH = "../client/input/inputs.csv"
LABELS_FILE_PATH = "../client/input/labels.csv"

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

def is_game_paused(telemetry):
    """Return True if the race is paused."""
    return (telemetry["cur_xpos"] == telemetry["prev_xpos"]) and 
           (telemetry["cur_ypos"] == telemetry["prev_ypos"]) and 
           (telemetry["cur_zpos"] == telemetry["prev_zpos"])

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

def get_current_race_telemetry():
    """Return the current race telemetry from Mario Kart Wii (PAL)."""
    telemetry = {
        "moving_dir": get_data_point(BASE_ADDRS["dir_base"], 0x61, "u8", deref=True),
        "steering_dir": get_data_point(BASE_ADDRS["dir_base"], 0x3C, "u8", deref=True),

        "cur_lap": get_data_point(BASE_ADDRS["lap_base"], 0x111, "u8", deref=True),
        "cur_lap_completion": get_data_point(BASE_ADDRS["lap_base"], 0xF8, "f32", deref=True),
        
        "cur_xpos": get_data_point(BASE_ADDRS["pos_base"], 0x40 + 0x0, "f32", deref=True),
        "cur_ypos": get_data_point(BASE_ADDRS["pos_base"], 0x40 + 0x4, "f32", deref=True),
        "cur_zpos": get_data_point(BASE_ADDRS["pos_base"], 0x40 + 0x8, "f32", deref=True),

        "prev_xpos": get_data_point(BASE_ADDRS["pos_base"], 0x40 - 0x160 + 0x0, "f32", deref=True),
        "prev_ypos": get_data_point(BASE_ADDRS["pos_base"], 0x40 - 0x160 + 0x4, "f32", deref=True),
        "prev_zpos": get_data_point(BASE_ADDRS["pos_base"], 0x40 - 0x160 + 0x8, "f32", deref=True),
    }

    return telemetry

def get_current_ctrls():
    """Gets the current ctrls from the WiiMote."""
    return {}

def write_data(data_list, filepath):
    """Writes list of data to the given file."""
    try:
        with open(filepath, "a+") as file:
            file.write(",".join(list(map(str, data_list))) + "\n")
    except Exception as e:
        print(e)

async def main():
    """Main function."""
    last_game_loaded = False
    last_game_id = None
    last_in_race = False
    last_is_paused = False
    frame_counter = 0

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
            frame_counter += 1
            gui.draw_text((10, 10), 0xffff0000, f"Frame: {frame_counter}")

            telemetry = get_current_race_telemetry()
            ctrls = get_current_ctrls()

            is_paused = is_game_paused(telemetry)
            if is_paused != last_is_paused:
                last_is_paused = is_paused

                if is_paused:
                    print("⏸️ Game paused.")
                else:
                    print("▶️ Game unpaused.")
                    await event.frameadvance()
                    continue

            write_data(telemetry.values(), INPUTS_FILE_PATH)
            write_data(ctrls.values(), LABELS_FILE_PATH)

        await event.frameadvance()


if __name__ == "__main__":
    await main()