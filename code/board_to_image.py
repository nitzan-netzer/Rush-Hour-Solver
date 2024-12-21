import os

import numpy as np
from moviepy.editor import ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont


def save_board_to_image(
    board, filename="board.png", scale=50, draw_grid=True, draw_letters=False
):
    """
    Save a board with symbols to an image file with optional grid and letters.

    Args:
        board: An object with `board` as a numpy 2D array.
        filename: Output image file name.
        scale: Size of each tile in pixels.
        draw_grid: Whether to draw a grid over the board.
    """
    # Map each symbol to a custom RGB color
    symbol_to_color = {
        "X": (255, 0, 0),  # Red
        "A": (144, 238, 144),  # Light Green
        "B": (255, 165, 0),  # Orange
        "C": (0, 255, 255),  # Cyan
        "D": (255, 182, 193),  # Pink
        "E": (0, 0, 139),  # Dark Blue
        "F": (0, 128, 0),  # Green
        "G": (50, 50, 50),  # Light Black (Grayish)
        "H": (245, 245, 220),  # Beige
        "I": (255, 255, 128),  # Light Yellow
        "J": (139, 69, 19),  # Saddle brown
        "K": (0, 255, 0),  # Green
        "O": (255, 255, 0),  # Yellow
        "P": (128, 0, 128),  # Purple
        "Q": (0, 0, 255),  # Blue
        "R": (0, 128, 128),  # Teal
    }

    rows, cols = board.board.shape
    rgb_data = np.zeros((rows, cols, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            symbol = board.board[r, c]
            rgb_data[r, c] = symbol_to_color.get(
                symbol, (128, 128, 128)
            )  # Gray for unknown

    img = Image.fromarray(rgb_data, "RGB")

    # Resize image to scale up the view
    if scale > 1:
        img = img.resize((cols * scale, rows * scale), Image.NEAREST)

    draw = ImageDraw.Draw(img)

    # Draw the grid if enabled
    if draw_grid:
        for x in range(0, cols * scale + 1, scale):
            draw.line((x, 0, x, rows * scale), fill=(0, 0, 0), width=1)
        for y in range(0, rows * scale + 1, scale):
            draw.line((0, y, cols * scale, y), fill=(0, 0, 0), width=1)
    # Highlight the edge to the right of (2, 5) with white color
    edge_row = 2 * scale  # Top of row 2 in pixels
    edge_col = (5 + 1) * scale  # Right edge of column 5 in pixels
    draw.line(
        (edge_col, edge_row, edge_col, edge_row + scale), fill=(255, 255, 255), width=5
    )
    if draw_letters:
        # Load default font
        try:
            font = ImageFont.truetype("arial.ttf", scale // 2)
        except:
            font = ImageFont.load_default()

        # Draw the letters in the center of each tile
        for r in range(rows):
            for c in range(cols):
                symbol = board.board[r, c]
                x = c * scale + scale // 2
                y = r * scale + scale // 2
                draw.text((x, y), symbol, fill=(0, 0, 0), font=font, anchor="mm")

    img.save(filename)
    # print(f"Board saved to {filename}")


def save_board_to_video(board, sol, frame_folder, video_name, draw_letters=False):
    """
    Generate a video from the frames
    """
    os.makedirs(frame_folder, exist_ok=True)

    # Save the initial board state
    save_board_to_image(board, f"{frame_folder}/frame_0.png", draw_letters=draw_letters)

    frame_count = 1
    # Apply each move in sol1
    for move in sol:
        symbol = move[0]
        car = board.get_vehicle_by_symbol(symbol)
        direction = move[1]
        times = move[2]
        for _ in range(int(times)):
            board.move_vehicle(car, direction)
            save_board_to_image(
                board,
                f"{frame_folder}/frame_{frame_count}.png",
                draw_letters=draw_letters,
            )
            frame_count += 1

    # Generate a video from the frames
    frame_files = [f"{frame_folder}/frame_{i}.png" for i in range(frame_count)]
    clip = ImageSequenceClip(frame_files, fps=4)  # Adjust FPS as needed
    clip.write_videofile(video_name, codec="libx264")

    print(f"Video saved as {video_name}")
