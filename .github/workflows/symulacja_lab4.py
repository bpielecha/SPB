import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

print("Zapisuję pliki w:", os.getcwd())

# Predefined patterns for Conway's Game of Life (uint8 dtype to ensure compatibility with Pillow)
PATTERNS = {
    "block": np.array([[1, 1], [1, 1]], dtype=np.uint8),
    "beehive": np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=np.uint8),
    "loaf": np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.uint8),
    "blinker": np.array([[1, 1, 1]], dtype=np.uint8),
    "toad": np.array([[0, 1, 1, 1], [1, 1, 1, 0]], dtype=np.uint8),
    "pulsar": np.array([
        [0,0,0,1,1,1,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,1,1,0,0,0,0,1],
        [1,0,0,0,0,0,1,1,0,0,0,0,1],
        [1,0,0,0,0,0,1,1,0,0,0,0,1],
        [0,0,0,1,1,1,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,0,0,0,1,1,1,0],
        [1,0,0,0,0,0,1,1,0,0,0,0,1],
        [1,0,0,0,0,0,1,1,0,0,0,0,1],
        [1,0,0,0,0,0,1,1,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,0,0,0,1,1,1,0]
    ], dtype=np.uint8),
    "r_pentomino": np.array([[0,1,1],[1,1,0],[0,1,0]], dtype=np.uint8)
}

class GameOfLife:
    def __init__(self, rows=50, cols=50):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=np.uint8)

    def initialize_pattern(self, name: str, top_left: tuple[int, int] = (0, 0)):
        r0, c0 = top_left
        if name == "gosper_glider_gun":
            gun = np.zeros((9, 36), dtype=np.uint8)
            cells = [
                (5,1),(5,2),(6,1),(6,2),(3,13),(3,14),(4,12),(5,11),(5,12),(6,11),
                (6,15),(7,11),(7,15),(8,14),(9,13),(1,25),(2,23),(2,25),(3,21),
                (3,22),(4,21),(4,22),(5,21),(5,22),(6,23),(6,25),(7,25),(3,35),
                (3,36),(4,35),(4,36)
            ]
            for rr, cc in cells:
                if 0 <= rr < 9 and 0 <= cc < 36:
                    gun[rr, cc] = 1
            pr, pc = gun.shape
            self.grid[r0:r0+pr, c0:c0+pc] = gun
        else:
            pattern = PATTERNS.get(name)
            if pattern is None:
                raise ValueError(f"Pattern '{name}' not found.")
            pr, pc = pattern.shape
            if r0+pr > self.rows or c0+pc > self.cols:
                raise ValueError("Pattern exceeds grid bounds.")
            self.grid[r0:r0+pr, c0:c0+pc] = pattern

    def randomize(self, density: float = 0.5):
        self.grid = (np.random.random((self.rows, self.cols)) < density).astype(np.uint8)

    def count_neighbors(self) -> np.ndarray:
        neighbors = np.zeros_like(self.grid)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbors += np.roll(np.roll(self.grid, dr, axis=0), dc, axis=1)
        return neighbors

    def step(self):
        nbrs = self.count_neighbors()
        birth = (self.grid == 0) & (nbrs == 3)
        survive = (self.grid == 1) & ((nbrs == 2) | (nbrs == 3))
        self.grid[...] = 0
        self.grid[birth | survive] = 1

    def run(self, iterations: int) -> list[np.ndarray]:
        frames = []
        for _ in range(iterations):
            frames.append(self.grid.copy())
            self.step()
        return frames

    def run_until_stable(self, min_iterations: int = 100, max_iterations: int = 10000) -> list[np.ndarray]:
        frames = []
        for i in range(max_iterations):
            frames.append(self.grid.copy())
            self.step()
            if i >= min_iterations and np.array_equal(self.grid, frames[-2]):
                break
        return frames


def animate(frames: list[np.ndarray], interval: int = 200, save_path: str | None = None):
    fig, ax = plt.subplots()
    img = ax.imshow(frames[0], interpolation='nearest', cmap='binary', vmin=0, vmax=1)
    ax.axis('off')

    def update(i):
        img.set_data(frames[i])
        return (img,)

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)

    if save_path:
        fps = 1000 / interval
        writer = animation.PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    game = GameOfLife(50, 50)

    # 2) Three still lifes
    game.grid.fill(0)
    game.initialize_pattern('block', (1, 1))
    game.initialize_pattern('beehive', (1, 10))
    game.initialize_pattern('loaf', (1, 20))
    frames2 = game.run(5)
    animate(frames2, save_path='sim2.gif')

    # 3) Two oscillators for at least 5 full cycles
    game.grid.fill(0)
    game.initialize_pattern('blinker', (1, 1))
    game.initialize_pattern('toad', (1, 10))
    frames3 = game.run(15)
    animate(frames3, save_path='sim3.gif')

    # 4) Gosper glider gun, show at least 3 gliders
    game.grid.fill(0)
    game.initialize_pattern('gosper_glider_gun', (1, 1))
    frames4 = game.run(100)
    animate(frames4, save_path='sim4.gif')

    # 5) R-pentomino until stable or 100 iterations
    game.grid.fill(0)
    game.initialize_pattern('r_pentomino', (25, 25))
    frames5 = game.run_until_stable(min_iterations=100)
    animate(frames5, save_path='sim5.gif')
