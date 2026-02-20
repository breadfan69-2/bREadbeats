import sys

TARGET = "87d99ae6c14ded4c22eaadb7411d423f6469dc31"

def main() -> int:
    if len(sys.argv) < 2:
        return 1
    todo_path = sys.argv[1]
    with open(todo_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    with open(todo_path, "w", encoding="utf-8", newline="") as file:
        for line in lines:
            if line.startswith(f"pick {TARGET} "):
                line = line.replace("pick", "reword", 1)
            file.write(line)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
