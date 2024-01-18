#!/usr/bin/python3

# Generate 1024 x 1024 matrix
# 3 columns * 1024 = 3072

x = 512
y = 512
vals = x * y

val = 0

writer = open("square.mtx", "w")
writer.write("%%MatrixMarket matrix coordinate real general\n")
writer.write("%\n")
writer.write("%\n")
writer.write(str(x) + " " + str(y) + " " + str(vals) + "\n")
for i in range(x):
    #writer.write(str(i+1) + "" + str(val) + "\n")
    #val += 1
    for j in range(y):
        writer.write(str(i+1) + " " + str(j+1) + " " + str(val) + "\n")
        val += 1
writer.close()

