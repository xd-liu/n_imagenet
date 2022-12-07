def main(fn):
    with open(fn, "r") as f:
        lines = f.readlines()
    
    acc_list = []
    for line in lines:
        words = line.strip("\n").split(" ")
        if words[0] == "Total":
            acc_list.append(float(words[-1]))
    
    print(max(acc_list))

if __name__ == "__main__":
    for name in ["./checkpoint/train-DiST/2022-11-27_17:39:37.log",
                    "./checkpoint/train-EH/2022-11-27_20:25:17.log",
                    "./checkpoint/train-STS/2022-11-27_20:22:57.log"
                    ]:
        print(name)
        main(name)
        
