from dolphin import memory

BASE_ADDR = 0x80000000

def is_game_loaded():
    """Check if a valid game is running in Dolphin and return its ID."""
    try:
        game_id_str = "".join(
            chr(memory.read_u8(BASE_ADDR + offset))
            for offset in range(6)
        )
    except Exception as e:
        print(e)
        return False, None

    if len(game_id_str) == 6 and game_id_str.isalnum():
        return True, game_id_str
    else:
        return False, None

if __name__ == "__main__":
    loaded, game_id_str = is_game_loaded()
    if loaded:
        print(f"✅ Game currently loaded: {game_id_str}")
    else:
        print("❌ No game loaded.")