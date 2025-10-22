def normalise_inputs(inputs: list) -> list:
    """Given a list with all inputs, normalise it to a set format"""
    normalised_inputs = [{
        "pos_x": round(row["pos_x"], 4),
        "pos_y": round(row["pos_y"], 4),
        "pos_z": round(row["pos_z"], 4),
        "speed": round(row["speed"], 4),
        "accel": round(row["accel"], 4),
        "lap":   round(row["lap"], 4),
    } for row in inputs]
    return normalised_inputs

def normalise_labels(labels: list) -> list:
    """Given a data file with all labels, normalise it to a set format"""
    normalised_labels = [{
        "2":           row["2"], 
        "1":           row["1"],
        "PLUS":        row["PLUS"],
        "DPAD_UP":     row["DPAD_UP"],
        "DPAD_DOWN":   row["DPAD_DOWN"],
        "DPAD_LEFT":   row["DPAD_LEFT"],
        "DPAD_RIGHT":  row["DPAD_RIGHT"],
        "STEER":       round((row["STEER"] / 7) - 1, 4)
    } for row in labels]
    return normalised_labels
