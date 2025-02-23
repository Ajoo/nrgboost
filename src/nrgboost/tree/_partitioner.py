# Provides specialized JITed implementations
import numpy as np
from numba import njit
from numba.extending import is_jitted


def maximize(objective_function, jit=True):
    if jit and not is_jitted(objective_function):
        objective_function = njit(objective_function)

    # @njit
    def gain_function(left, right, parent):
        return objective_function(*left) + objective_function(*right) \
            - objective_function(*parent)

    return njit(gain_function) if jit else gain_function


def minimize(objective_function, jit=True):
    if jit and not is_jitted(objective_function):
        objective_function = njit(objective_function)

    # @njit
    def gain_function(left, right, parent):
        return objective_function(*parent) - \
            objective_function(*left) - objective_function(*right) \

    return njit(gain_function) if jit else gain_function


def st(constraints, jit=True):
    num_constraints = len(constraints)
    constraints = tuple(c if is_jitted(c) else njit(c) for c in constraints)

    # @njit
    def mask_function(left, right, _):
        mask = constraints[0](*left)
        mask *= constraints[0](*right)
        if num_constraints > 1:
            for constraint in constraints[1:]:
                mask *= constraint(*left)
                mask *= constraint(*right)

        return mask

    return njit(mask_function) if jit else mask_function


def make_find_dim_split(objective_func, constraint_func=None):
    # TODO: make sure objective and constraints are jitted

    if not constraint_func:
        @njit
        def _find_dim_split(left, right, parent):
            gain = objective_func(left, right, parent)
            idx = np.argmax(gain)
            return idx, gain.flat[idx]

    else:
        @njit
        def _find_dim_split(left, right, parent):
            gain = objective_func(left, right, parent)
            constraints = constraint_func(left, right, parent)

            gain[~constraints] = -np.inf
            idx = np.argmax(gain)
            return idx, gain[idx]  # gain.flat[idx]

    return _find_dim_split  # njit(_find_dim_split)


def make_find_dim_split_from_marginals(num_data, objective_func, constraint_func=None, sort_func=None, jit_unordered=False):

    _find_dim_split = make_find_dim_split(objective_func, constraint_func)
    if sort_func is not None and not is_jitted(sort_func):
        sort_func = njit(sort_func)

    if num_data == 2:
        @njit
        def _find_dim_split_ordered(m0, m1):
            M0 = np.cumsum(m0)
            M1 = np.cumsum(m1)
            idx, gain = _find_dim_split(
                (M0[:-1], M1[:-1]),
                (M0[-1] - M0[:-1], M1[-1] - M1[:-1]),
                (M0[-1], M1[-1])
            )
            return gain, idx + 1

        if sort_func is None:
            @njit
            def _find_dim_split_unordered(m0, m1):
                M0 = np.sum(m0)
                M1 = np.sum(m1)
                idx, gain = _find_dim_split(
                    (M0 - m0, M1 - m1),
                    (m0, m1),
                    (M0, M1)
                )
                return gain, idx
        else:
            # @njit
            def _find_dim_split_unordered(m0, m1):
                values = sort_func(m0, m1)
                sort_index = np.argsort(values)  # , kind='mergesort')

                gain, best = _find_dim_split_ordered(
                    m0[sort_index], m1[sort_index])
                return gain, sort_index[:best]

            if jit_unordered:
                _find_dim_split_unordered = njit(_find_dim_split_unordered)

    elif num_data == 1:
        @njit
        def _find_dim_split_ordered(m0):
            M0 = np.cumsum(m0)
            idx, gain = _find_dim_split(
                (M0[:-1],),
                (M0[-1] - M0[:-1],),
                (M0[-1],)
            )
            return gain, idx + 1
        if sort_func is None:
            @njit
            def _find_dim_split_unordered(m0):
                M0 = np.sum(m0)
                idx, gain = _find_dim_split(
                    (M0 - m0,),
                    (m0,),
                    (M0,)
                )
                return gain, idx
        else:
            @njit
            def _find_dim_split_unordered(m0):
                values = sort_func(m0)
                sort_index = np.argsort(values)
                idx, gain = _find_dim_split_ordered(
                    m0[sort_index])
                # if sort_index.ndim <= 1:  # only 1 ordering
                return gain, sort_index[:idx]
                # else:  # multiple orderings, best is chosen
                #     split_index = np.unravel_index(
                #         split_index, sort_index.shape)
                #     split_index = sort_index[(
                #         slice(split_index[0]+1), *split_index[1:])]

    else:
        raise NotImplementedError

    return _find_dim_split_ordered, _find_dim_split_unordered


def make_find_best_split(num_data, ordered, objective_func, constraint_func=None, sort_func=None, jit_unordered=False, with_feature_frac=False):
    find_split_ordered, find_split_unordered = make_find_dim_split_from_marginals(
        num_data, objective_func, constraint_func=constraint_func, sort_func=sort_func, jit_unordered=jit_unordered)

    if num_data != 2:
        raise NotImplementedError

    num_dims = ordered.size
    num_dims_ordered = np.sum(ordered)
    num_dims_unordered = num_dims - num_dims_ordered

    if num_dims_unordered == 0:
        find_split_unordered = find_split_ordered
        sort_func = None

    # TODO: also check if only unordered dimensions are binary
    # in that case call find_split_ordered
    if sort_func is None:
        if with_feature_frac:
            @njit(parallel=False)
            def _find_best_split(dims, cardinalities, marginals0, marginals1):
                num_dims = dims.size
                index = np.empty(num_dims + 1, dtype=np.intp)
                index[0] = 0
                index[1:] = np.cumsum(cardinalities)

                gains = np.empty(num_dims, dtype=np.float64)
                split_indexes = np.empty(num_dims, dtype=np.int16)

                for i in range(num_dims):
                    start = index[i]
                    stop = index[i+1]
                    if cardinalities[i] == 1:
                        gains[i] = -np.inf
                        split_indexes[i] = -1
                    elif ordered[dims[i]]:
                        gains[i], split_indexes[i] = find_split_ordered(
                            marginals0[start:stop], marginals1[start:stop])
                    else:
                        gains[i], split_indexes[i] = find_split_unordered(
                            marginals0[start:stop], marginals1[start:stop])

                dim_idx = np.argmax(gains)
                return gains[dim_idx], dims[dim_idx], split_indexes[dim_idx]
        else:
            @njit(parallel=False)
            def _find_best_split(cardinalities, marginals0, marginals1):
                index = np.empty(num_dims + 1, dtype=np.intp)
                index[0] = 0
                index[1:] = np.cumsum(cardinalities)

                gains = np.empty(num_dims, dtype=np.float64)
                split_indexes = np.empty(num_dims, dtype=np.int16)

                for i in range(num_dims):
                    start = index[i]
                    stop = index[i+1]
                    if cardinalities[i] == 1:
                        gains[i] = -np.inf
                        split_indexes[i] = -1
                    elif ordered[i]:
                        gains[i], split_indexes[i] = find_split_ordered(
                            marginals0[start:stop], marginals1[start:stop])
                    else:
                        gains[i], split_indexes[i] = find_split_unordered(
                            marginals0[start:stop], marginals1[start:stop])

                dim_idx = np.argmax(gains)
                return gains[dim_idx], dim_idx, split_indexes[dim_idx]
    else:
        if with_feature_frac:
            if jit_unordered:
                @njit
                def __find_best_split(dims, cardinalities, marginals0, marginals1):
                    num_dims = dims.size
                    index = np.empty(num_dims + 1, dtype=np.intp)
                    index[0] = 0
                    index[1:] = np.cumsum(cardinalities)

                    gain = -np.inf
                    dim_idx = -1
                    ordered_dims = ordered[dims]
                    split_index_dims = 1 if not np.any(
                        ordered_dims) else np.max(cardinalities[~ordered_dims])
                    # there is at least one ordered dim
                    split_indexes = np.empty(split_index_dims, dtype=np.int16)
                    split_indexes[0] = -1
                    num_indexes = 1

                    for i in range(num_dims):
                        start = index[i]
                        stop = index[i+1]
                        if cardinalities[i] == 1:
                            continue
                        elif ordered_dims[i]:
                            candidate_gain, ordered_split_index = find_split_ordered(
                                marginals0[start:stop], marginals1[start:stop])
                            if candidate_gain > gain:
                                gain = candidate_gain
                                dim_idx = i
                                split_indexes[0] = ordered_split_index
                                num_indexes = 1
                            # split_indexes[i] = np.array([split_index])
                        else:
                            candidate_gain, unordered_split_index = find_split_unordered(
                                marginals0[start:stop], marginals1[start:stop])
                            if candidate_gain > gain:
                                gain = candidate_gain
                                dim_idx = i
                                num_indexes = unordered_split_index.size
                                split_indexes[:num_indexes] = unordered_split_index

                    return gain, dims[dim_idx], split_indexes[:num_indexes]

                def _find_best_split(dims, cardinalities, marginals0, marginals1):
                    gain, dim_idx, split_indexes = __find_best_split(
                        dims, cardinalities, marginals0, marginals1)
                    return (
                        gain,
                        dim_idx,
                        split_indexes[0] if ordered[dim_idx] else split_indexes
                    )
            else:
                def _find_best_split(dims, cardinalities, marginals0, marginals1):
                    num_dims = dims.size
                    index = np.empty(num_dims + 1, dtype=np.intp)
                    index[0] = 0
                    index[1:] = np.cumsum(cardinalities)

                    gains = np.empty(num_dims, dtype=np.float64)
                    split_indexes = []

                    for i in range(num_dims):
                        start = index[i]
                        stop = index[i+1]
                        if cardinalities[i] == 1:
                            gains[i] = -np.inf
                            si = -1
                        elif ordered[dims[i]]:
                            gains[i], si = find_split_ordered(
                                marginals0[start:stop], marginals1[start:stop])
                        else:
                            gains[i], si = find_split_unordered(
                                marginals0[start:stop], marginals1[start:stop])
                        split_indexes.append(si)

                    dim_idx = np.argmax(gains)
                    return gains[dim_idx], dims[dim_idx], split_indexes[dim_idx]
        else:
            if jit_unordered:
                @njit
                def __find_best_split(cardinalities, marginals0, marginals1):
                    index = np.empty(num_dims + 1, dtype=np.intp)
                    index[0] = 0
                    index[1:] = np.cumsum(cardinalities)

                    gain = -np.inf
                    dim_idx = -1
                    split_indexes = np.empty(
                        np.max(cardinalities[~ordered]), dtype=np.int16)  # there is at least one ordered dim
                    split_indexes[0] = -1
                    num_indexes = 1

                    for i in range(num_dims):
                        start = index[i]
                        stop = index[i+1]
                        if cardinalities[i] == 1:
                            continue
                        elif ordered[i]:
                            candidate_gain, ordered_split_index = find_split_ordered(
                                marginals0[start:stop], marginals1[start:stop])
                            if candidate_gain > gain:
                                gain = candidate_gain
                                dim_idx = i
                                split_indexes[0] = ordered_split_index
                                num_indexes = 1
                            # split_indexes[i] = np.array([split_index])
                        else:
                            candidate_gain, unordered_split_index = find_split_unordered(
                                marginals0[start:stop], marginals1[start:stop])
                            if candidate_gain > gain:
                                gain = candidate_gain
                                dim_idx = i
                                num_indexes = unordered_split_index.size
                                split_indexes[:num_indexes] = unordered_split_index

                    return gain, dim_idx, split_indexes[:num_indexes]

                def _find_best_split(cardinalities, marginals0, marginals1):
                    gain, dim_idx, split_indexes = __find_best_split(
                        cardinalities, marginals0, marginals1)
                    return (
                        gain,
                        dim_idx,
                        split_indexes[0] if ordered[dim_idx] else split_indexes
                    )
            else:
                def _find_best_split(cardinalities, marginals0, marginals1):
                    index = np.empty(num_dims + 1, dtype=np.intp)
                    index[0] = 0
                    index[1:] = np.cumsum(cardinalities)

                    gains = np.empty(num_dims, dtype=np.float64)
                    split_indexes = []

                    for i in range(num_dims):
                        start = index[i]
                        stop = index[i+1]
                        if cardinalities[i] == 1:
                            gains[i] = -np.inf
                            si = -1
                        elif ordered[i]:
                            gains[i], si = find_split_ordered(
                                marginals0[start:stop], marginals1[start:stop])
                        else:
                            gains[i], si = find_split_unordered(
                                marginals0[start:stop], marginals1[start:stop])
                        split_indexes.append(si)

                    dim_idx = np.argmax(gains)
                    return gains[dim_idx], dim_idx, split_indexes[dim_idx]

    return _find_best_split
