# you can solve dictionary mimimizing problem here

n = 6


def solve():
    T = []
    s = "ACDBCB"
    arr = s
    s_t = arr[::-1]
    temp = list(s)
    for i in xrange(n):
        s = "".join(temp)
        s_t = s[::-1]
        print temp
        if s_t > s:
            T.append(temp.pop(0))
        else:
            T.append(temp.pop())
    print T
    
if __name__ == '__main__':
    solve()
