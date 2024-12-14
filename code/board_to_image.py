"""
This module provides a function to save a board to an image file.
"""
from PIL import Image, ImageDraw
import numpy as np

def save_board_to_image(board, filename="board.png", scale=50, draw_grid=True):
    # Map each symbol to a custom RGB color
    # Choose any distinct colors you like
    symbol_to_color = {
        "X": (255, 0, 0),       # RedCar (special case)

        "B": (0, 0, 255),       # Blue car
        "G": (0, 128, 0),       # Green car
        "O": (255, 165, 0),     # Orange car
        "C": (0, 255, 255),     # Cyan car
        "M": (255, 0, 255),     # Magenta car
        "W": (255, 255, 255),   # White car

        "y": (255, 255, 0),     # Yellow truck
        "p": (128, 0, 128),     # Purple truck
    }

    rows, cols = board.board.shape
    rgb_data = np.zeros((rows, cols, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            symbol = board.board[r, c]
            # Get the color if symbol is known, else gray for unknown
            rgb_data[r, c] = symbol_to_color.get(symbol, (128, 128, 128))

    img = Image.fromarray(rgb_data, 'RGB')

    # Resize image to scale up the view
    if scale > 1:
        img = img.resize((cols * scale, rows * scale), Image.NEAREST)

    # Optionally draw a grid
    if draw_grid:
        draw = ImageDraw.Draw(img)
        # Draw vertical grid lines
        for x in range(0, cols * scale + 1, scale):
            draw.line((x, 0, x, rows * scale), fill=(0, 0, 0), width=1)
        # Draw horizontal grid lines
        for y in range(0, rows * scale + 1, scale):
            draw.line((0, y, cols * scale, y), fill=(0, 0, 0), width=1)

    img.save(filename)
    print(f"Board saved to {filename}")
