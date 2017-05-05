import itertools
import math
import copy
import random
import heapq
import numpy as np
from sklearn.naive_bayes import GaussianNB

'''
 prints out the board
'''

def printBoard(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            print(str(board[i][j]) + "   ",end='')
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

ps = possibleSolutions(4)


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



def availMoves(gameBoard):
    moves = [[(i,j) for(j,x) in enumerate(row) if x==0] for (i,row) in enumerate(gameBoard)]
    moves = [j for i in moves for j in i]
    #print(moves)
    return moves

def minMax2(gameBoard, player, nodes_expanded=1):
    gameover = gameOver(gameBoard)

    if gameover[1] == -1:
        return (.5, nodes_expanded, [gameBoard], 0.5)
    if gameover[1] == 1:
        return (1, nodes_expanded, [gameBoard], 0)
    if gameover[1] == 2:
        return (0, nodes_expanded, [gameBoard], 1)

    nextplayer = 2 if player == 1 else 1
    scores = [minMax2(succ, nextplayer, nodes_expanded) for succ in [move(gameBoard, m[0], m[1], player) for m in availMoves(gameBoard)]]

    best_path = None
    val = 0
    winner = -1
    if player == 1:
        val = -float("inf")
    else:
        val = float("inf")

    for s in scores:
        mmval, ne, bp, wn = s
        if player == 1:
            if val <= mmval:
                best_path = bp
                val = mmval
                winner = wn
        else:
            val = min(val, mmval)
            if val >= mmval:
                best_path = bp
                val = mmval
                winner = wn

    nodes_expanded += sum([s[1] for s in scores])
    return (val, nodes_expanded, best_path + [gameBoard], winner)
    

def miniMax(gameBoard, player, nodes_expanded=1):
    gameover = gameOver(gameBoard)

    if gameover[1] == -1:
        return (gameBoard, .5, nodes_expanded, -1)
    if gameover[1] == 1:
        return (gameBoard, 1, nodes_expanded, 1)
    if gameover[1] == 2:
        return (gameBoard, 0, nodes_expanded, 2)

    nextplayer = 2 if player == 1 else 1
    scores = [(succ, miniMax(succ, nextplayer, nodes_expanded)) for succ in [move(gameBoard, m[0], m[1], player) for m in availMoves(gameBoard)]]

    board = []
    winner = -1
    val = 0
    if player == 1:
        val = -float('inf')
    else:
        val = float('inf')
        
    for succ, s in scores:
        brd, v, ne, wn = s
        if player == 1:
            if v >= val:
                val = v
                board = succ
                winner = wn
        else:
            if v <= val:
                val = v
                board = succ
                winner = wn
    
    nodes_expanded += sum([s[1][2] for s in scores])
    print("%s: %s" % ('1' if val == 0 else '2' if val == 1 else '0', ' '.join(["%d %d %d %d" % (z[0], z[1],z[2],z[3]) for z in board])))
    return (board, val, nodes_expanded, winner)

    

def alphaBetazzzzz(gameBoard, alpha, beta, depth, turn, nodes_expanded=1):
    if(depth == 0 or gameOver(gameBoard)[0]):
        return [staticEval(gameBoard), nodes_expanded, [gameBoard]]

    #printBoard(gameBoard)
    best_path = None
    if turn ==1:
        v = -float('inf')
        for m in availMoves(gameBoard):
            succ = move(gameBoard,m[0],m[1],turn)
            t, ne, bp = alphaBeta(succ,alpha,beta,depth-1,2, nodes_expanded)
            nodes_expanded = ne +1
            if v <= t:
                v = t
                best_path = bp
            alpha = max(alpha,v)
            if beta <= alpha:
                break
        return (v, nodes_expanded, best_path + [gameBoard])
    else:
        v = float('inf')
        for m in availMoves(gameBoard):
            succ = move(gameBoard,m[0],m[1],turn)
            
            #v = min(v,alphaBeta(succ,alpha,beta,depth-1,1))
            t, ne, bp = alphaBeta(succ,alpha,beta,depth-1,1, nodes_expanded)
            nodes_expanded = ne + 1
            #v = min(v, t)
            if v >= t:
                v = t
                best_path = bp
            
            beta = min(beta,v)
            if beta <= alpha:
                break
        return (v, nodes_expanded, best_path + [gameBoard])



def alphaBeta(gameBoard, alpha, beta, depth, turn, nodes_expanded=1):
    if(depth == 0 or gameOver(gameBoard)[0]):
        return [gameBoard, staticEval(gameBoard), nodes_expanded]

    if turn == 1:
        v = -float('inf')
        for m in availMoves(gameBoard):
            succ = move(gameBoard, m[0], m[1], turn)
            board, t, ne = alphaBeta(succ, alpha, beta, depth-1, 2, nodes_expanded)
            nodes_expanded = ne+1
            v = max(v, t)
            alpha = max(alpha, v)
            if beta <= alpha:
                break
        return (succ, v, nodes_expanded)
    else:
        v = float('inf')
        for m in availMoves(gameBoard):
            succ = move(gameBoard, m[0], m[1], turn)
            board, t, ne = alphaBeta(succ, alpha, beta, depth-1, 1, nodes_expanded)
            nodes_expanded = ne+1
            v = min(v, t)
            beta = min(beta, v)
            if beta <= alpha:
                break
        return (succ, v, nodes_expanded)

def alphaBetaNB(classifier, gameBoard, alpha, beta, depth, turn, nodes_expanded=1):
    if(depth == 0 or gameOver(gameBoard)[0]):
        return [gameBoard, staticEval(gameBoard), nodes_expanded]

    moves = availMoves(gameBoard)
    #print(moves)
    #a = [[b for b in board for b in b] for board in moves]
    #print(a)
    #q = classifier.predict()
    #print(q)
    #z = [(classifier.predict([b for b in board for b in b]), board) for board in moves]
    #print(z)
    #scored = heapq.heapify(z)
    #print("gb:", gameBoard)
    #printBoard(gameBoard)
    #print(moves)
    scored = [(classifier.predict([[b for b in board for b in b]]), board) for board in [move(gameBoard, m[0], m[1], turn) for m in moves]]
    #print(scored)
    #heapq.heapify(scored)
    #print("sc:", list(scored))
    scored.sort(key=lambda a: a[0])
    #print("sc:", list(scored))
    scored = [a[1] for a in scored]
    
    
    if turn == 1:
        v = -float('inf')
        best = scored[-1]
        for succ in scored:
            #succ = move(gameBoard, m[0], m[1], turn)
            board, t, ne = alphaBetaNB(classifier, succ, alpha, beta, depth-1, 2, nodes_expanded)
            nodes_expanded = ne+1
            v = max(v, t)
            alpha = max(alpha, v)
            if beta <= alpha:
                best = succ
                break
        return (best, v, nodes_expanded)
    else:
        v = float('inf')
        best = scored[-1]
        for succ in scored:
            #succ = move(gameBoard, m[0], m[1], turn)
            board, t, ne = alphaBetaNB(classifier, succ, alpha, beta, depth-1, 1, nodes_expanded)
            nodes_expanded = ne+1
            v = min(v, t)
            beta = min(beta, v)
            if beta <= alpha:
                best = succ
                break
        return (best, v, nodes_expanded)



        
def testCase(gameBoard, player=1):
    #print("board:")
    #printBoard(gameBoard)
    #print()

    print("minimax:")
    val, nodes, states, winner = minMax2(gameBoard, player)
    #board, val, nodes, winner = miniMax(gameBoard, player)
    #states.reverse()
    #for s in states:
    #    printBoard(s)
    #printBoard(states[0])
    #print ("val: " + str(val))
    print ("winner: " + str(winner))
    print ("nodes: " + str(nodes))
    print()

    print("alphabeta:")
    #print(minMax(gameBoard,1))
    #print(alphaBeta(gameBoard,-float('inf'),float('inf'),6,1))
    board, val, nodes = alphaBeta(gameBoard, -float('inf'), float('inf'), 6, player)
    printBoard(board)
    #for s in states:
        #print (s, staticEval(s))
    #    printBoard(s)
    print ("root val: " + str(val))
    print ("nodes: " + str(nodes))
    print("-"*20)
    print()
    print()

def getRandomBoard():
    turns = random.randint(1, 16)
    board = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    for i in range(turns):
        next = availMoves(board)
        m = random.choice(next)
        board = move(board, m[0], m[1], 1 if i % 2 == 0 else 2)
        
    return [board, turns%2 + 1]

def generateTrainingSet():
    while True:
        board, turn = getRandomBoard()
        b, val, np, wn = miniMax(board, turn)

def main():
    # get row and col from user
    #user_input = input("enter in row and col space seperated: ")
    #if(len(user_input) < 3):
    #    print("error in input")
    #row = int(user_input[0])
    #col = int(user_input[2])
    #row = col = 4

    #init gameBoard
    ##gameBoard = []
    #for i in range(row):
    #    gameBoard.append([])
    #    for j in range(col):
    #        gameBoard[i].append(0)
    #possibleSolutions(len(gameBoard))
    #testCase(gameBoard)
    #printBoard(gameBoard)
    #print(minMax(gameBoard,0,1))

    #printBoard(getRandomBoard()[0])
    #printBoard(getRandomBoard()[0])
    #printBoard(getRandomBoard()[0])
    #printBoard(getRandomBoard()[0])
    
    #generateTrainingSet()
    #return
    

    gameBoard = [[2,0,0,2],
                 [2,1,0,1],
                 [2,0,0,1],
                 [0,0,0,1]]

    gameBoard = [[1,2,0,0],
                 [1,0,2,0],
                 [2,0,0,1],
                 [2,0,0,1]]
    print("alphabeta:")
    #print(minMax(gameBoard,1))
    #print(alphaBeta(gameBoard,-float('inf'),float('inf'),6,1))
    board, val, nodes = alphaBeta(gameBoard, -float('inf'), float('inf'), 6, 1)
    printBoard(board)
    #for s in states:
        #print (s, staticEval(s))
    #    printBoard(s)
    print ("root val: " + str(val))
    print ("nodes: " + str(nodes))
    print("-"*20)
    print()
    print()

    
    print("alphabeta NB:")
    tdata = []
    labels = []
    with open("training.txt") as f:
        for line in f:
            k, v = line.split(":")
            #labels.append(.5 if k == '0' else 0 if k == '1' else 1)
            labels.append(int(k))
            v = v.strip().split(' ')
            #tdata.append([v[0:4], v[4:8], v[8:12], v[12:16]])
            tdata.append([int(z) for z in v])
    classifier = GaussianNB()
    #print(labels[:10])
    #print(tdata[:10])
    #X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    #Y = np.array([1, 1, 1, 2, 2, 2])
    #classifier.fit(X, Y)
    
    #classifier.fit(tdata[:10], labels[:10])
    classifier.fit(tdata, labels)

    board, val, nodes = alphaBetaNB(classifier, gameBoard, -float('inf'), float('inf'), 6, 1)
    printBoard(board)
    #for s in states:
        #print (s, staticEval(s))
    #    printBoard(s)
    print ("root val: " + str(val))
    print ("nodes: " + str(nodes))
    print("-"*20)

    
    #miniMax(gameBoard, 1)
    #return
    #testCase(gameBoard)

    return

    gameBoard = [[1,2,0,0],
                 [1,0,2,0],
                 [2,0,0,1],
                 [2,0,0,1]]
    #testCase(gameBoard)
    
    gameBoard = [[1,2,2,0],
                 [1,1,2,0],
                 [0,0,0,0],
                 [2,1,0,0]]
    #testCase(gameBoard)
    gameBoard = [[2,0,2,0],
                 [1,0,0,1],
                 [1,2,0,0],
                 [0,0,0,1]]
    #testCase(gameBoard, 2)





main()


