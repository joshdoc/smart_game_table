def main():
    count = 0
    print("Press Enter to increment the counter. Type 'q' and press Enter to quit.\n")

    while True:
        user_input = input("Press Enter (or type 'q' to quit): ")
        if user_input.lower() == 'q':
            print("Exiting program. Goodbye!")
            break
        count += 1
        print(f"Count: {count}")

if __name__ == "__main__":
    main()
