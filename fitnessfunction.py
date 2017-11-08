def fitness(board):
    """Calculates the fitness for a particular board and updates the
    board's fitness accordingly"""
    squarePenalty = 1
    groupPenalty = 2
    groupFlag = False
    score = 0
    
    for i in range (0, board.nonogram_size):
        squareCount = 0
        groupCount = 0
        squares = 0
        groups = 0
        for y in board.row_numbers[i]:
            groupCount++
            squareCount = squareCount + y
            
        for j in range (0, board.nonogram_size):
            if (board.grid[i][j] == 1):
                squares++
                if (!groupFlag):
                    groups++
                groupFlag = True
            if (board.grid[i][j] == -1):
                if (groupFlag):
                    groupFlag = False
        squares = abs(squares - squareCount)
        groups = abs(groups  - groupCount)
        score = score + (squarePenalty * squares + groupPenalty * groups)

    for i in range (0, board.nonogram_size):
        squareCount = 0
        groupCount = 0
        squares = 0
        groups = 0
        for y in board.column_numbers[i]:
            groupCount++
            squareCount = squareCount + y

        for j in range (0, board.nonogram_size):
            if (board.grid[j][i] == 1):
                square++
                if (!groupFlag):
                    groups++
                groupFlag = True
            if (board.grid[j][i] == -1):
                if (groupFlag):
                    groupFlag = False

        squares = abs(squares - squareCount)
        groups = abs(groups - groupCount)
        score = score + (squarePenalty * squares + groupPenalty * groups)
        
    
    board.fitness = score
