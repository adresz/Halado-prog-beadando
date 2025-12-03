def step_function(net):
    return 1 if net >= 0 else 0

def print_header():
    print("\n" + "="*110)
    print(" " * 40 + "3 VÁLTOZÓS AND PERCEPTRON TANÍTÁS")
    print("="*110)
    print(f"{'Iter':<5} {'x1':<3} {'x2':<3} {'x3':<3} {'d':<3} {'net':>9} {'y':<3} {'e':<4} "
          f"{'w1':>9} {'w2':>9} {'w3':>9} {'b':>9}  {'Státusz'}")
    print("-" * 110)

def print_row(iter_num, x1, x2, x3, d, net, y, e, w1, w2, w3, b, correct):
    status = "Correct" if correct else "Incorrect"
    color = "\033[92m" if correct else "\033[91m"
    reset = "\033[0m"
    print(f"{iter_num:<5} {x1:<3} {x2:<3} {x3:<3} {d:<3} {net:9.4f} {y:<3} {e:+4} "
          f"{w1:9.4f} {w2:9.4f} {w3:9.4f} {b:9.4f}  {color}{status}{reset}")

# =============== ADATOK ===============
data = [
    [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0],
    [1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]
]

