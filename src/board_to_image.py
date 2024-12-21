import numpy as np
from moviepy.editor import ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont

car_colors = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
truck_colors = ["O", "P", "Q", "R"]
letter_to_color = {
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


def generate_board_image(board, scale: int = 50, draw_letters: bool = False) -> Image:
    """
    Generate an in-memory PIL image for a given board state.

    Args:
        board: An object with `board` as a numpy 2D array.
        scale: Size of each tile in pixels.
        draw_letters: Whether to draw letters on the board tiles.

    Returns:
        A PIL.Image object.
    """
    rows, cols = board.board.shape
    rgb_data = np.zeros((rows, cols, 3), dtype=np.uint8)
    gray_default = (128, 128, 128)
    for r in range(rows):
        for c in range(cols):
            letter = board.board[r, c]
            rgb_data[r, c] = letter_to_color.get(letter, gray_default)

    img = Image.fromarray(rgb_data, "RGB")

    if scale > 1:
        img = img.resize((cols * scale, rows * scale), Image.NEAREST)

    draw = ImageDraw.Draw(img)

    # Draw grid lines
    for x in range(0, cols * scale + 1, scale):
        draw.line((x, 0, x, rows * scale), fill=(0, 0, 0), width=1)
    for y in range(0, rows * scale + 1, scale):
        draw.line((0, y, cols * scale, y), fill=(0, 0, 0), width=1)

    if draw_letters:
        try:
            font = ImageFont.truetype("arial.ttf", scale // 2)
        except OSError:
            font = ImageFont.load_default()

        for r in range(rows):
            for c in range(cols):
                letter = board.board[r, c]
                x = c * scale + scale // 2
                y = r * scale + scale // 2
                draw.text((x, y), letter, fill=(0, 0, 0), font=font, anchor="mm")

    # Highlight the edge to the right of (2, 5) with white color
    edge_row = 2 * scale  # Top of row 2 in pixels
    edge_col = (5 + 1) * scale  # Right edge of column 5 in pixels
    white = (255, 255, 255)
    cords = [(edge_col, edge_row), (edge_col, edge_row + scale)]
    draw.line(cords, fill=white, width=5)

    return img


def save_board_to_image(
    board, filename: str, scale: int = 50, draw_letters: bool = False
):
    """
    Save a board with letters to an image file with optional grid and letters.

    Args:
        board: An object with `board` as a numpy 2D array.
        filename: Output image file name.
        scale: Size of each tile in pixels.
        draw_grid: Whether to draw a grid over the board.
    """
    img = generate_board_image(board, scale, draw_letters)
    img.save(filename)

    print(f"Board saved to {filename}")


def save_board_to_video(
    board, sol: tuple[str], video_name: str, draw_letters: bool = False, fps: int = 4
):
    """
    Generate a video directly from board states without saving intermediate frames.

    Args:
        board: The board object.
        sol: A list of moves to apply to the board.
        video_name: Output video file name.
        draw_letters: Whether to draw letters on the board tiles.
        fps: Frames per second for the output video.
    """
    frames = []

    # Generate the initial board state as an image
    img = generate_board_image(board, draw_letters=draw_letters)
    frames.append(img)

    # Apply each move in the solution and capture frames
    for move in sol:
        letter = move[0]
        car = board.get_vehicle_by_letter(letter)
        direction = move[1]
        times = move[2]
        for _ in range(int(times)):
            board.move_vehicle(car, direction)
            img = generate_board_image(board, draw_letters=draw_letters)
            frames.append(img)

    # Convert the frames to an ImageSequenceClip
    clip = ImageSequenceClip([np.array(frame) for frame in frames], fps=fps)
    clip.write_videofile(video_name, codec="libx264", logger=None)

    print(f"Video saved as {video_name}")
