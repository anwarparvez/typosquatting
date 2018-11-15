

def save_txt(file_name,content):
    file = open(file_name, 'w+')
    file.write(content)
    file.close()
    return file
