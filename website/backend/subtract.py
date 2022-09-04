import sys

n=len(sys.argv)
f = open('output-subtract.txt','w')
sys.stdout = f
a=int(sys.argv[1])

b=int(sys.argv[2])

print(a-b)