import setup_path  # NOQA
if True:
    from environments.card_parkings import main as parking_main
    from environments.cards_generator import main as generator_main
    from environments.cards_original import main as original_main


def main():
    print("\nRunning tests for parking module:")
    parking_main()
    print("\nRunning tests for original module:")
    original_main()
    print("\nRunning tests for generator module:")
    generator_main()
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
