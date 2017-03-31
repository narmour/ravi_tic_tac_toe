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
        #print("invalid move")
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

        print("p1 wins: " + str(p1_wins))
        print("p2 wins: " + str(p2_wins))

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
    print()

    return unique

def availMoves(gameBoard):
    moves = [[(i,j) for(j,x) in enumerate(row) if x==0] for (i,row) in enumerate(gameBoard)]
    moves = [j for i in moves for j in i]
    print(moves)
    return moves

def minMax(gameBoard,num_expanded,turn):
    val = 0
    # if its a leaf node
    if(gameOver(gameBoard)[0]):
        print("game over, winner: "+ str(gameOver(gameBoard)[1]))
        printBoard(gameBoard)
        return staticEval(gameBoard)
    
    # if its a max node
    if (turn==1):
        val = -float('inf')
        print("yo")
    elif(turn==2):
        val = float('inf')


    for m in availMoves(gameBoard):
        n = 2 if turn==1 else 1
        succ = move(gameBoard,m[0],m[1],turn)
        printBoard(succ)
        if(turn ==1):
            val = max(val,minMax(succ,num_expanded+1,n))
            #print("max: " + str(val))
        else:
            val = min(val,minMax(succ,num_expanded+1,n))
            #print("mmin: " + str(val))
    return val






def testCase(gameBoard):
    '''
    gameBoard = move(gameBoard,0,0,1)
    gameBoard = move(gameBoard,0,1,2)
    gameBoard = move(gameBoard,0,2,1)
    gameBoard = move(gameBoard,0,3,2)
    gameBoard = move(gameBoard,1,0,2)
    gameBoard = move(gameBoard,1,1,1)
    gameBoard = move(gameBoard,2,0,1)
    gameBoard = move(gameBoard,3,0,2)
    gameBoard = move(gameBoard,2,1,1)
    gameBoard = move(gameBoard,2,2,2)
    gameBoard = move(gameBoard,3,3,1)
    '''
    '''
    gameBoard = move(gameBoard,0,2,1)
    gameBoard = move(gameBoard,1,1,1)
    gameBoard = move(gameBoard,1,3,1)
    gameBoard = move(gameBoard,2,2,1)
    gameBoard = move(gameBoard,3,0,1)
    '''
    gameBoard = move(gameBoard,0,0,1)
    gameBoard = move(gameBoard,1,1,1)
    gameBoard = move(gameBoard,2,2,1)
    gameBoard = move(gameBoard,3,3,1)
    gameBoard = move(gameBoard,3,1,1)
    printBoard(gameBoard)
    gameOver(gameBoard)
    #print(minMax(gameBoard,0,1))

def main():
    # get row and col from user
    user_input = input("enter in row and col space seperated: ")
    if(len(user_input) < 3):
        print("error in input")
    row = int(user_input[0])
    col = int(user_input[2])

    #init gameBoard
    gameBoard = []
    for i in range(row):
        gameBoard.append([])
        for j in range(col):
            gameBoard[i].append(0)
    possibleSolutions(len(gameBoard))
    testCase(gameBoard)
    #printBoard(gameBoard)
    #print(minMax(gameBoard,0,1))













main()


