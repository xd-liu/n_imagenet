def main(fn):
    with open(fn, "r") as f:
        lines = fn.readlines()
    
    acc_list = []
    for line in lines:
        words = line.stripe("\n").split(" ")
        if words[0] == "Total":
            acc_list.append(float(words[-1]))
    
    print(max(acc_list))

if __name__ == "__main__":
    for name in ["./checkpoint/train-DiST/2022-11-28_16:13:05.log",
    "./checkpoint/train-EH/2022-11-28_16:13:46.log",
    ]