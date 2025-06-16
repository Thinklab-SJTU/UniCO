import jittor as jt


def get_random_problems(batch_size, node_cnt):

    ################################
    # "tmat" type
    ################################

    scaler = 1000000
    int_min = 0
    int_max = 1000000
    
    problems = jt.randint(low=int_min, high=int_max, shape=(batch_size, node_cnt, node_cnt))
    # shape: (batch, node, node)
    problems[:, jt.arange(node_cnt), jt.arange(node_cnt)] = 0

    while True:
        old_problems = problems.clone()

        problems = (problems[:, :, None, :] + problems[:, None, :, :].transpose(2,3)).min(dim=3)
        # shape: (batch, node, node)

        if (problems == old_problems).all():
            break

    # Scale
    scaled_problems = problems.float() / scaler

    return scaled_problems
    # shape: (batch, node, node)


def load_single_problem_from_file(filename, node_cnt=None, scaler=1e6):

    ################################
    # "tmat" type
    ################################
    if node_cnt is not None:
        problem = jt.empty(size=(node_cnt, node_cnt), dtype=jt.long)
    # shape: (node, node)

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except Exception as err:
        print(str(err))

    line_cnt = 0
    for line in lines:
        linedata = line.split()

        if linedata[0].startswith(('TYPE', 'DIMENSION', 'EDGE_WEIGHT_TYPE', 'EDGE_WEIGHT_FORMAT', 'EDGE_WEIGHT_SECTION', 'EOF')):
            continue

        integer_map = map(int, linedata)
        integer_list = list(integer_map)

        if node_cnt is None:
            node_cnt = len(integer_list)
            problem = jt.empty((node_cnt, node_cnt)).long()

        problem[line_cnt] = jt.Var(integer_list).long()
        line_cnt += 1

    # Diagonals to 0
    problem[jt.arange(node_cnt), jt.arange(node_cnt)] = 0

    # Scale
    scaled_problem = problem.float() / scaler

    return scaled_problem
    # shape: (node, node)
