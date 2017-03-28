
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
    if(board[i][j] ==0):
        board[i][j] = player
    else:
        print("invalid move")


'''
returns true if it finds a player has won
returns false if no player has own
'''
def gameOver(board):
    rowsTaken1 = []
    colsTaken1 = []

    rowsTaken2 = []
    colsTaken2 = []

    for i in range(len(board)):
        for j in range(len(board[i])):
            if(board[i][j] == 1):
                rowsTaken1.append(i)
                colsTaken1.append(j)
            elif(board[i][j] == 2):
                rowsTaken2.append(i)
                colsTaken2.append(j)


    if (len(set(rowsTaken1)) == len(board) and len(set(colsTaken1)) == len(board)):
        return True
    elif (len(set(rowsTaken2)) == len(board) and len(set(colsTaken2)) == len(board)):
        return True
    else:
        return False
    


'''
determines the value of the board from the first players perspective(1) defined
as follows: if its a win for player 1, the value is 100, if its a win for player 2,
then the value is -100. Otherwise, it is the number of ways in which player 1 can win
minus the number of ways in which player 2 can win.
'''
def staticEval(gameBoard):
    return 100


def testCase(gameBoard):
    #should return true
    move(gameBoard,0,0,1)
    move(gameBoard,1,2,1)
    move(gameBoard,2,3,1)
    move(gameBoard,3,1,1)
    printBoard(gameBoard)
    print(str(gameOver(gameBoard)))

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
    printBoard(gameBoard)


    testCase(gameBoard)











main()


