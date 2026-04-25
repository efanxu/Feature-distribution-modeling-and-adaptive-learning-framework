import numpy as np


def find_extremes(series):
    extremes = []
    n = len(series)
    for i in range(1, n - 1):
        prev_val = series[i - 1]
        curr_val = series[i]
        next_val = series[i + 1]
        if curr_val > prev_val and curr_val > next_val:
            extremes.append((i, 'max'))
        elif curr_val < prev_val and curr_val < next_val:
            extremes.append((i, 'min'))
    return extremes


def find_four_alternate_extremes(extremes):
    if len(extremes) < 4:
        return None
    reversed_extremes = extremes[::-1]
    first_type = reversed_extremes[0][1]
    if first_type == 'max':
        target_sequence = ['max', 'min', 'max', 'min']
    else:
        target_sequence = ['min', 'max', 'min', 'max']
    collected = []
    current_target_idx = 0
    for point in reversed_extremes:
        if point[1] == target_sequence[current_target_idx]:
            collected.append(point)
            current_target_idx += 1
            if current_target_idx == 4:
                break
    if len(collected) != 4:
        return None
    return collected[3][0]


def find_segment(series):
    if len(series) < 5:
        return None
    extremes = find_extremes(series)
    if len(extremes) < 4:
        return None
    start_pos = find_four_alternate_extremes(extremes)
    if start_pos is None:
        return None
    return series[start_pos:]


def dtw_distance(X, Y, window=2, best_so_far=np.inf):
    X = np.array(X)
    Y = np.array(Y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    assert X.shape[1] == Y.shape[1]

    n, m = X.shape[0], Y.shape[0]
    dist_matrix = np.sqrt(np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2))

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)
        min_row_val = np.inf
        for j in range(j_start, j_end):
            cost = dist_matrix[i - 1, j - 1]
            min_prev = min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
            dtw[i, j] = cost + min_prev
            if dtw[i, j] < min_row_val:
                min_row_val = dtw[i, j]

        if min_row_val >= best_so_far:
            return np.inf

    return dtw[n, m]


def find_similar_segment(X, Y, YY, extend_len, max_search_steps=None, dynamic_threshold=np.inf, window=2):
    X_len = len(X)
    Y_len = len(Y)

    if X_len > Y_len - X_len:
        raise ValueError("X长度应小于Y长度的一半，以确保能找到两段不重叠的Y的子序列")

    min_distance = np.inf
    best_start_index = None

    if max_search_steps is not None:
        start_limit = max(0, Y_len - max_search_steps)
    else:
        start_limit = 0

    for start in range(Y_len - X_len, start_limit - 1, -1):
        segment = Y[start:start + X_len]
        distance = dtw_distance(X, segment, window=window, best_so_far=min_distance)
        if distance < min_distance:
            min_distance = distance
            best_start_index = start

    if min_distance > dynamic_threshold:
        print("警告：触发熔断机制，使用镜像扩展")
        next_segment = X[-extend_len:][::-1]
        similar_segment = X
        return similar_segment, next_segment

    similar_segment = Y[best_start_index:best_start_index + X_len]
    next_segment = YY[best_start_index + X_len:best_start_index + X_len + extend_len]
    return similar_segment, next_segment
