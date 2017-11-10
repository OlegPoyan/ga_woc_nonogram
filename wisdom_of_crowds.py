def wisdom_of_crowds(boards):
    size = boards[0].nonogram_size
    adj_matrix = [[0 for x in range (0, size)] for y in range (0, size)]
    for pop_size in range (0, len(boards)):
        for i in range (0, size):
            for j in range (0, size):
                if (boards[pop_size].grid[i][j] == 1):
                    adj_matrix[i][j] += 1
    for i in range (0, size):
        for j in range (0, size):
            adj_matrix[i][j] /= size
    return adj_matrix

def wisdom_create_board(adj_matrix, threshold):
    size = len(adj_matrix)
    board = [[0 for x in range (0, size)] for y in range (0, size)]
    for i in range (0, size):
        for j in range (0, size):
            if (adj_matrix[i][j] <= threshold):
                board[i][j] = 0
            if (adj_matrix[i][j] > 1-threshold):
                board[i][j] = 1
            else:
                board[i][j] = randint(0, 1)

    return board


def wisdom_apply_mask (adj_matrix, boards, threshold):
    size = boards[0].nonogram_size
    for pop_size in range (0, len(boards)):
        for i in range (0, size):
            for j in range (0, size):
                if (adj_matrix[i][j] <= threshold):
                    boards[pop_size].grid[i][j] = 0
                if (adj_matrix[i][j] > 1-threshold):
                    boards[pop_size].grid[i][j] = 1
                else:
                    #DO NOTHING

    return boards
