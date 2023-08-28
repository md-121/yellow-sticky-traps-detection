# https://stackoverflow.com/questions/58383814/how-to-divide-an-image-into-evenly-sized-overlapping-if-needed-tiles
# NOT USED
import numpy as np


def tile(h, w, tile_width=None, tile_height=None, window_size=100):
    np.seterr(divide='ignore', invalid='ignore')

    if not tile_width:
        tile_width = window_size

    if not tile_height:
        tile_height = window_size

    wTile = tile_width
    hTile = tile_height

    if tile_width > w or tile_height > h:
        raise ValueError("tile dimensions cannot be larger than origin dimensions")

    # Number of tiles
    nTilesX = np.uint8(np.ceil(w / wTile))
    nTilesY = np.uint8(np.ceil(h / hTile))

    # Total remainders
    remainderX = nTilesX * wTile - w
    remainderY = nTilesY * hTile - h

    # Set up remainders per tile
    remaindersX = np.ones((nTilesX - 1, 1)) * np.uint16(np.floor(remainderX / (nTilesX - 1)))
    remaindersY = np.ones((nTilesY - 1, 1)) * np.uint16(np.floor(remainderY / (nTilesY - 1)))
    remaindersX[0:np.remainder(remainderX, np.uint16(nTilesX - 1))] += 1
    remaindersY[0:np.remainder(remainderY, np.uint16(nTilesY - 1))] += 1

    # Initialize array of tile boxes
    tiles = np.zeros((nTilesX * nTilesY, 4), np.uint16)

    k = 0
    x = 0
    for i in range(nTilesX):
        y = 0
        for j in range(nTilesY):
            tiles[k, :] = (x, y, hTile, wTile)
            k += 1
            if j < (nTilesY - 1):
                y = y + hTile - remaindersY[j]
        if i < (nTilesX - 1):
            x = x + wTile - remaindersX[i]

    return tiles


if __name__ == "__main__":
    tiles = tile(3456, 5184, 800, 1320)
    print(tiles)
