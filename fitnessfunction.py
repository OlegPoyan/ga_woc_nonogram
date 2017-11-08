def fitness(board):
    """Calculates the fitness for a particular board and updates the
    board's fitness accordingly"""
    squarePenalty = 1
    groupPenalty = 2
    groupFlag = False
    score = 0
    
    # checks row by row to get a score based on number of groups
    # and number of on squares needed compared to how many found
    for i in range (0, board.nonogram_size):
        squareCount = 0
        groupCount = 0
        squares = 0
        groups = 0
        for y in board.row_numbers[i]:
            groupCount+=1
            squareCount = squareCount + y
            
        for j in range (0, board.nonogram_size):
            if (board.grid[i][j] == 1):
                squares+=1
                if not groupFlag:
                    groups+=1
                groupFlag = True
            if (board.grid[i][j] == -1):
                if (groupFlag):
                    groupFlag = False
        squares = abs(squares - squareCount)
        groups = abs(groups  - groupCount)
        score = score + (squarePenalty * squares + groupPenalty * groups)

    # checks column by column to get a score based on number of groups
    # and number of on squares needed compared to how many found
    for i in range (0, board.nonogram_size):
        squareCount = 0
        groupCount = 0
        squares = 0
        groups = 0
        for y in board.column_numbers[i]:
            groupCount+=1
            squareCount = squareCount + y

        for j in range (0, board.nonogram_size):
            if (board.grid[j][i] == 1):
                square+=1
                if not groupFlag:
                    groups+=1
                groupFlag = True
            if (board.grid[j][i] == -1):
                if (groupFlag):
                    groupFlag = False

        squares = abs(squares - squareCount)
        groups = abs(groups - groupCount)
        score = score + (squarePenalty * squares + groupPenalty * groups)
        
    # updates the board fitness with the b
    board.fitness = score
