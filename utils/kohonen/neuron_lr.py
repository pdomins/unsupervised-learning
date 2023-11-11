
def fixed_learning_rate(curr_iter: int, lr: float = 0.1) -> float:
    return lr

def inv2iter_learning_rate(curr_iter: int) -> float:
    return 1 / (1 + curr_iter)

def inv2iter_flex_learning_rate(curr_iter: int, a: float = 1, b: float = 1, c: float = 0) -> float:
    return a / (b + curr_iter) + c