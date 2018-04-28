
def hasnan(x):
    return (x != x).any()

def parse_param(param):
    splitted = (x.split("=") for x in param)
    return {y[0]: y[1] for y in splitted}
