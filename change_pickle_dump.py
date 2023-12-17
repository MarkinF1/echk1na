import os
import pickle
from argparse import ArgumentParser


dump = None
args = None


def load_dump():
    global dump, args
    
    if os.path.exists(args.file):
        with open(args.file, "rb") as file:
            dump = pickle.load(file)
    else:
        print(f"Не нашел файл, отключение. NoSuchFileOrDirectory: {args.file}")
        exit(1)


def save_dump(dump, filename=None):
    global args
    
    i = True
    
    def print_save():
        nonlocal i
        
        if i:
            print("Сохранение...", end='\r')
            i = False
        else:
            print("Сохранение завершено.", end='\r')
    
    print_save()
    if filename is None:
        filename = args.file

    with open(args.file, "wb") as file:
        pickle.dump(dump, file)
    print_save()


def view():
    global dump, args
    
    for key, value in dump.items():
        print(f"Ключ: {key}\nКоличество объектов с таким ключом: {len(dump[key])}")
        print('-' * 40)
        

def delete():
    global dump, args
    
    key = (args.unit, args.direction)
    try:
        dump.pop(key)
        print(f"Ключ {key} успешно удален.")
        save_dump(dump)
    except KeyError:
        print(f"Ключ {key} не найден, отключение.")
    
def pop():
    global args, dump

    key = (args.unit, args.direction)
    try:
        part_dump = dump[key]
        print(f"Ключ {key} получен.")
        save_dump(part_dump, f"./pickle_u{args.unit}_d{args.direction}.pl")
    except KeyError:
        print(f"Ключ {key} не найден, отключение.")

def main():
    global dump, args
    
    
    functions = {"view": view,
                 "pop": pop,
                 "delete": delete
                }
                
    parser = ArgumentParser(description="Описание программы")

    parser.add_argument("-f", "--file", type=str, help="Файл для загрузки")
    parser.add_argument("-c", "--command", type=str, choices=["view", "delete", "pop"], help="Команда для запуска")
    parser.add_argument("-u", "--unit", type=int, default=None, help="Unit для удаления")
    parser.add_argument("-d", "--direction", type=int, default=None, help="Direction для удаления")
    
    args = parser.parse_args()
    load_dump()
    
    try:
        functions[args.command]()
    except KeyError:
        print("Команда не найдена, отключение.")
        exit(1)
        
        
if __name__ == "__main__":
    main()
    
