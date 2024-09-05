
import argparse
import subprocess
import matplotlib.pyplot as plt
import re


##########################
# Modify the following for different benchmark data
gridSizes = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
iterList = [iters for  iters in range(1, 1000, 100)]
averageOver = 10
##########################

parser = argparse.ArgumentParser()
parser.add_argument("benchmarkExecutable")
benchmarkExe = parser.parse_args().benchmarkExecutable

for gridSize in gridSizes:
    kernelTimes, cudaTimes, seqTimes = [], [], []
    for iters in iterList:
        kernelTime, cudaTime, seqTime = 0, 0, 0
        for i in range(averageOver):
            out = subprocess.check_output([benchmarkExe, str(gridSize[0]), str(gridSize[1]), str(iters)]).decode()
            times = re.findall("[0-9]+", out)
            kernelTime += int(times[0])/averageOver
            cudaTime += int(times[1])/averageOver
            seqTime += int(times[2])/averageOver
        kernelTimes.append(kernelTime)
        cudaTimes.append(cudaTime)
        seqTimes.append(seqTime)

    plt.clf()
    plt.plot(iterList, kernelTimes, "o-", color="red", label="Time to run kernel")
    plt.plot(iterList, cudaTimes, "o-", color="purple", label="Time to run full CUDA algorithm")
    plt.plot(iterList, seqTimes, "o-", color="blue", label="Time to run sequential algorithm")
    
    title = "{}x{} grid".format(gridSize[0], gridSize[1])
    plt.title(title)
    plt.legend()
    plt.xlabel("Number of iterations")
    plt.ylabel("Runtime (ms)")

    filename = "figures/runtimes_{}x{}.png".format(gridSize[0], gridSize[1])
    plt.savefig(filename)
    print("Saved graph for {}*{} grid".format(gridSize[0], gridSize[1]))






