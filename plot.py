import matplotlib.pylab as plt


def read_from_file():
    with open("generation_analysis.txt", "r") as file:
        lines = file.readlines()

    max_list, min_list, avg_list = [], [], []
    for line in lines:
        line_arr = line.split(" ")
        min_list.append(int(line_arr[0]))
        avg_list.append(float(line_arr[1]))
        max_list.append(int(line_arr[2]))

    return min_list, avg_list, max_list


if __name__ == "__main__":
    min_list, avg_list, max_list = read_from_file()
    plt.plot(min_list, color='green', label="min list")
    plt.plot(avg_list, color='red', label="average list")
    plt.plot(max_list, color='blue', label="max list")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()