import numpy as np

# corner_actions = 4 * 3 q* 5  # 4 corners, 3 possible moves, 5 possible build
#
# board_actions = 4 * 3 * 5 # 4 sides of 3 squares, 5 possible moves,

possible_row = possible_col = [-1, 0, 1]

move_id = {}
move_count = 0
board = np.zeros((5, 5))
for row in range(0, 5):
    for col in range(0, 5):
        loc = (row, col)
        for i in possible_row:
            for j in possible_col:
                try:
                    new_loc = (loc[0] + i, loc[1]+ j)
                    if new_loc[0] < 0 or new_loc[1] < 0:
                        continue
                    if new_loc[0] == loc[0] and new_loc[1] == loc[1]:
                        continue
                    try:
                        a = board[new_loc[0], new_loc[1]]
                    except:
                        continue
                    for y in possible_row:
                        for x in possible_col:
                            try:
                                build = (new_loc[0] + y, new_loc[1] + x)
                                if build[0] < 0 or build[1] < 0:
                                    continue
                                if build[0] == new_loc[0] and build[1] == new_loc[1]:
                                    continue
                                try:
                                    b = board[build[0], build[1]]
                                except:
                                    continue
                                move_id[f"[{row}, {col}]->[{new_loc[0]}, {new_loc[1]}]->[{build[0]}, {build[1]}]"] = move_count
                                move_count += 1
                            except:
                                continue
                except:
                    continue

