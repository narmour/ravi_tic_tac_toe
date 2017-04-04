import itertools
import math
import copy
'''
 prints out the board
'''

def printBoard(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            print(str(board[i][j]) + "   ",end='')
        print()

    print()


'''
board=gameBoard
i = row
j = col
player = 1 or 2

function returns board modified with [i][j] set to player if its empty
else does nothing
'''
def move(board,i,j,player):
    b = copy.deepcopy(board)
    if(b[i][j] ==0):
        b[i][j] = player
    else:
        print("invalid move")
        return board
    return b


'''
returns true if it finds a player has won
returns false if no player has own
'''
def gameOver(board):
    p1 = []
    p2 = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if(board[i][j] ==1):
                p1.append((i,j))
            elif(board[i][j] ==2):
                p2.append((i,j))
    ps = possibleSolutions(len(board))

    p1_p = list(itertools.combinations(p1,len(board)))
    for p in p1_p:
        if list(p) in ps:
            return(True,1)

    p2_p = list(itertools.combinations(p2,len(board)))
    for p in p2_p:
        if list(p) in ps:
            return (True,2)

    if len(p1) + len(p2) == len(board) * len(board[0]):
        return (True, -1)
    


    return (False,0)




'''
determines the value of the board from the first players perspective(1) defined
as follows: if its a win for player 1, the value is 100, if its a win for player 2,
then the value is -100. Otherwise, it is the number of ways in which player 1 can win
minus the number of ways in which player 2 can win.
'''
def staticEval(gameBoard):
    status = gameOver(gameBoard)
    if(status[0] and status[1] == 1):
        return 100
    elif(status[0] and status[1] ==2):
        return -100
    else:
        p1_wins = 0
        p2_wins = 0
        solutions = possibleSolutions(len(gameBoard))
        for s in solutions:
            possible1 =1
            possible2 =1
            for k in s:
                val = gameBoard[k[0]][k[1]]
                if(val ==2):
                    possible1 =0
                if(val ==1):
                    possible2 = 0
            if(possible1):
                p1_wins+=1
            if(possible2):
                p2_wins+=1

        #print("p1 wins: " + str(p1_wins))
        #print("p2 wins: " + str(p2_wins))

        return p1_wins - p2_wins
                
'''
    returns a list of all possible solutions to this game
    with row col  being filled
'''
def possibleSolutions(k):
    solutions = []
    rowsTaken = []
    colsTaken=[]
    permutes = list(itertools.permutations(range(k)))
    for p in range(len(permutes)):
        solutions.append([])
        rowsTaken.append([])
        colsTaken.append([])
        for i in range(k):
            for j in range(k):
                if(i not in rowsTaken[p] and permutes[p][j] not in colsTaken[p]):
                    solutions[p].append((i,permutes[p][j]))
                    rowsTaken[p].append(i)
                    colsTaken[p].append(permutes[p][j])

    #get rid of duplicates
    unique = []
    [unique.append(s) for s in solutions if s not in unique]

    #print("possible solutions")
    #[print(s) for s in unique]
    #print()

    return unique

def availMoves(gameBoard):
    moves = [[(i,j) for(j,x) in enumerate(row) if x==0] for (i,row) in enumerate(gameBoard)]
    moves = [j for i in moves for j in i]
    #print(moves)
    return moves


def minMax(gameBoard,turn, nodes_expanded=1):
    # if its a leaf node
    #best_path = [gameBoard]
    gameover = gameOver(gameBoard)
    #if gameOver(gameBoard)[1] == 1 or gameOver(gameBoard)[1] == 2:
        #print("game over, winner: "+ str(gameOver(gameBoard)[1]))
        #print("nodes expanded: ",nodes_expanded)
        #printBoard(gameBoard)
    #    return (staticEval(gameBoard), nodes_expanded, [gameBoard])
    #if gameover[0]:
    #if gameover[1] == -1:
    #    return (.5, nodes_expanded, [gameBoard])
    if gameover[1] == 1:
        return (1, nodes_expanded, [gameBoard])
    if gameover[1] == 2:
        return (0, nodes_expanded, [gameBoard])
    
    # if its a max node
    val = 0
    if (turn==1):
        val = -float('inf')
    elif(turn==2):
        val = float('inf')


    best_path = None
    best = [0, None]
    n = 2 if turn == 1 else 1
    #printBoard(gameBoard)
    #print(availMoves(gameBoard))
    sval = 0
    for m in availMoves(gameBoard):
        #nodes_expanded = nodes_expanded +1
        succ = move(gameBoard,m[0],m[1],turn)
        #printBoard(succ)
        if turn == 1:
            mmval, ne, bp = minMax(succ, n, nodes_expanded)
            nodes_expanded = ne + 1
            val = max(val,mmval)
            #best_path = bp
            best, best_path = [[val, gameBoard], bp] if val >= best[0] or best[1] == None else [best, best_path]
        else:
            #val = min(val,minMax(succ,n))
            mmval, ne, bp = minMax(succ, n, nodes_expanded)
            nodes_expanded = ne + 1
            val = min(val,mmval)
            #best_path = bp
            best, best_path = [[val, gameBoard], bp] if val <= best[0] or best[1] == None else [best, best_path]
            #best = [val, gameBoard] if val >= best[0] or best[1] == None else best
        #best = playerbest(succ, val, n, best)
    #printBoard(gameBoard)
    if best_path == None: # a draw happened
        return (0, nodes_expanded, ['draw'])
    return (val, nodes_expanded, best_path + [best[1]])

def minMax(gameBoard, player, nodes_expanded=1):
    gameover = gameOver(gameBoard)

    if gameover[1] == -1:
        return (.5, nodes_expanded, [gameBoard])
    if gameover[1] == 1:
        return (1, nodes_expanded, [gameBoard])
    if gameover[1] == 2:
        return (0, nodes_expanded, [gameBoard])

    nextplayer = 2 if player == 1 else 1
    scores = [minMax(succ, nextplayer, nodes_expanded) for succ in [move(gameBoard, m[0], m[1], player) for m in availMoves(gameBoard)]]

    best_path = None
    val = 0
    if player == 1:
        val = -float("inf")
    else:
        val = float("inf")

    for s in scores:
        mmval, ne, bp = s
        if player == 1:
            if val <= mmval:
                best_path = bp
                val = mmval
        else:
            val = min(val, mmval)
            if val >= mmval:
                best_path = bp
                val = mmval

    nodes_expanded += sum([s[1] for s in scores])
    return (val, nodes_expanded, best_path + [gameBoard])
    

def testCase(gameBoard):
    gameBoard = move(gameBoard,0,0,2)
    gameBoard = move(gameBoard,0,3,2)
    gameBoard = move(gameBoard,1,0,2)
    gameBoard = move(gameBoard,1,1,1)
    gameBoard = move(gameBoard,1,3,1)
    gameBoard = move(gameBoard,2,0,2)
    gameBoard = move(gameBoard,2,3,1)
    gameBoard = move(gameBoard,3,3,1)
    '''
    test case 1
    gameBoard = move(gameBoard,0,0,2)
    gameBoard = move(gameBoard,0,3,2)
    gameBoard = move(gameBoard,1,0,2)
    gameBoard = move(gameBoard,1,1,1)
    gameBoard = move(gameBoard,1,3,1)
    gameBoard = move(gameBoard,2,0,2)
    gameBoard = move(gameBoard,2,3,1)
    gameBoard = move(gameBoard,3,3,1)
    '''

    '''
    test case 2
    gameBoard = move(gameBoard,0,0,1)
    gameBoard = move(gameBoard,0,1,2)
    gameBoard = move(gameBoard,0,2,2)
    gameBoard = move(gameBoard,1,0,1)
    gameBoard = move(gameBoard,1,1,1)
    gameBoard = move(gameBoard,1,2,2)
    gameBoard = move(gameBoard,3,0,2)
    gameBoard = move(gameBoard,3,1,1)
    '''

    '''
    test case 3
    gameBoard = move(gameBoard,0,0,2)
    gameBoard = move(gameBoard,0,2,2)
    gameBoard = move(gameBoard,1,0,1)
    gameBoard = move(gameBoard,1,3,1)
    gameBoard = move(gameBoard,2,0,1)
    gameBoard = move(gameBoard,2,1,2)
    gameBoard = move(gameBoard,3,3,1)
    '''

    gameBoard = [[1,2,1,2],
                 [2,1,0,0],
                 [1,0,0,0],
                 [2,0,0,0]]

    print(staticEval(gameBoard))
    
    gameBoard = [[2,0,2,0],
                 [1,0,0,1],
                 [1,2,0,0],
                 [0,0,0,1]]

    gameBoard = [[2,0,0,2],
                 [2,1,0,1],
                 [2,0,0,1],
                 [0,0,0,1]]

    
    printBoard(gameBoard)

    val, nodes, states = minMax(gameBoard,1)
    states.reverse()
    print (val, nodes)
    print(states)
    for s in states:
        print (s, staticEval(s))
        printBoard(s)


def main():
    # get row and col from user
    #user_input = input("enter in row and col space seperated: ")
    #if(len(user_input) < 3):
    #    print("error in input")
    #row = int(user_input[0])
    #col = int(user_input[2])
    row = col = 4

    #init gameBoard
    gameBoard = []
    for i in range(row):
        gameBoard.append([])
        for j in range(col):
            gameBoard[i].append(0)
    #possibleSolutions(len(gameBoard))
    testCase(gameBoard)
    #printBoard(gameBoard)
    #print(minMax(gameBoard,0,1))













main()


