import setup_path  # NOQA
if True:
    from GUI.rush_hour_ui import main as rush_hour_ui_main
    from GUI.visualizer import main as visualizer_main



def main():
    print("\nRunning rush_hour_ui module:")
    visualizer_main()
    print("\nRunning visualizer module:")
    rush_hour_ui_main()
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
