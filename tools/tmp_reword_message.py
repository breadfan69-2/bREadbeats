import sys

NEW_MESSAGE = "Repository hygiene updates"

def main() -> int:
    if len(sys.argv) < 2:
        return 1
    message_path = sys.argv[1]
    with open(message_path, "w", encoding="utf-8", newline="") as file:
        file.write(NEW_MESSAGE + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
