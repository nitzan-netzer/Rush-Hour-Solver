# generate_and_split_cards.py
import setup_path
import os
from environments.board_random import BoardRandom
from cards_generator import cards_generator, save
from environments.board import Board
from utils.config import BOARD_SIZE


def generate_boards_for_size(size: int, num_cards: int = 4000, num_cars: int = 11, num_trucks: int = 4, num_step: int = 10, threshold: int = 1):
    """
    Generate a batch of random boards for a given board size.
    """
    print(f"\n🔧 Generating {num_cards} cards for board size {size}x{size}...")
    BOARD_SIZE = size  # override config
    boards = cards_generator(
        num_cards, num_cars, num_trucks, num_step, threshold, shuffle=True)
    return boards


def sort_and_split(boards):
    """
    Sort boards by heuristic and split into four difficulty tiers.
    """
    for b in boards:
        b.heuristic = b.get_heuristic()

    sorted_boards = sorted(boards, key=lambda b: b.heuristic)
    n = len(sorted_boards)
    q1 = n // 4
    q2 = n // 2
    q3 = 3 * n // 4
    return {
        "easy": sorted_boards[:q1],
        "medium": sorted_boards[q1:q2],
        "hard": sorted_boards[q2:q3],
        "expert": sorted_boards[q3:]
    }


def save_all_tiers(tiers, board_size: int, base_dir="database/generated"):
    """
    Save all difficulty tiers into separate JSON files.
    """
    os.makedirs(base_dir, exist_ok=True)
    for tier, tier_boards in tiers.items():
        filename = f"{base_dir}/{board_size}x{board_size}_{tier}.json"
        print(f"✅ Saving {tier} ({len(tier_boards)} boards) → {filename}")
        Board.save_multiple_boards(tier_boards, filename)


def run_all_sizes():
    """
    Generate + split + save for all sizes
    """
    for size in [6, 8, 10]:
        boards = generate_boards_for_size(size)
        tiers = sort_and_split(boards)
        save_all_tiers(tiers, size)


if __name__ == "__main__":
    run_all_sizes()
