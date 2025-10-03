from dolphin import memory

RACE_ADDR_PAL = 0x809C18F8
IN_RACE_OFFSET = 0x2B

def is_in_race():
    """Return True if Mario Kart Wii (PAL) is currently racing."""
    try:
        ptr_val = memory.read_u32(RACE_ADDR_PAL)
        stage = memory.read_u8(ptr_val + IN_RACE_OFFSET)
        return stage == 1
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    if is_in_race():
        print("🏁 In race.")
    else:
        print("🕹️ Not in race.")