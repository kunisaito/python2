#this scipts think of the problem where we want to pay money as few coins as we can.

coin = [1,5,10,50,100,500]
# you can consider the amount of money you want to pay



def solve(money):

    count = []
    for i in xrange(6):
        count.append(money/coin[5-i])
        money = money - count[i]*coin[5-i]
        

    for i in xrange(6):
        print count[i]

if __name__ == '__main__':
    money = 24492
    solve(money)
        
