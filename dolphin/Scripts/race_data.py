from dolphin import memory

BASE_ADDRS = {
    "dir_base": 0x809BD70C,
    "lap_base": 0x809BD730,
    "pos_base": 0x809C2EF8,
    "time_base": 0x809BD730 
}

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

def get_current_race_data():
    """Return the current race data from Mario Kart Wii (PAL)."""
    return {
        "moving_dir": get_data_point(BASE_ADDRS["dir_base"], 0x61, "u8", deref=True),
        "steering_dir": get_data_point(BASE_ADDRS["dir_base"], 0x3C, "u8", deref=True),

        "cur_lap": get_data_point(BASE_ADDRS["lap_base"], 0x111, "u8", deref=False),
        "max_lap": get_data_point(BASE_ADDRS["lap_base"], 0x112, "u8", deref=False),
        "cur_lap_completion": get_data_point(BASE_ADDRS["lap_base"], 0xF8, "f32", deref=False),
        "max_lap_completion": get_data_point(BASE_ADDRS["lap_base"], 0xFC, "f32", deref=False),

        "cur_xpos": get_data_point(BASE_ADDRS["pos_base"], 0x40 + 0x0, "f32", deref=False),
        "cur_ypos": get_data_point(BASE_ADDRS["pos_base"], 0x40 + 0x4, "f32", deref=False),
        "cur_zpos": get_data_point(BASE_ADDRS["pos_base"], 0x40 + 0x8, "f32", deref=False),
        "prev_xpos": get_data_point(BASE_ADDRS["pos_base"], 0x40 - 0x160 + 0x0, "f32", deref=False),
        "prev_ypos": get_data_point(BASE_ADDRS["pos_base"], 0x40 - 0x160 + 0x4, "f32", deref=False),
        "prev_zpos": get_data_point(BASE_ADDRS["pos_base"], 0x40 - 0x160 + 0x8, "f32", deref=False),

        "minutes": get_data_point(BASE_ADDRS["time_base"], 0x1B9, "u8", deref=False),
        "seconds": get_data_point(BASE_ADDRS["time_base"], 0x1BA, "u8", deref=False),
        "third_seconds": get_data_point(BASE_ADDRS["time_base"], 0x1BC, "u8", deref=False)
    }

if __name__ == "__main__":
    data = get_current_race_data()
    print(data)