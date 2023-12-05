def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()