DATA_FILE_PATH = "../client/input/data.csv"
sample_data = "EXAMPLE DATA\n"

def write_data(data_to_write):
    """Writes data to the given file."""
    try:
        with open(DATA_FILE_PATH, "a+") as file:
            file.write(data_to_write + "\n")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    print("Writing data...")
    write_data(sample_data)
    print("Done.")