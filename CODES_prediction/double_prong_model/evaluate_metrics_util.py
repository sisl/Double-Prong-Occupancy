'''
Code from https://github.com/BenQLange/Occupancy-Grid-Metrics
'''

import numpy as np
import math

def MSE(X_test, X_hat, start_time = 0):

    mse_per_frame = np.mean(((X_test[:, start_time:] - X_hat[:, start_time:])**2), axis=(2,3))
    mse_score = np.mean( (X_test[:, start_time:] - X_hat[:, start_time:])**2 )

    return mse_score, mse_per_frame

def ImageSimilarityMetric(X_test, X_hat, start_time = 0):

    num_samples, num_times,_,_ = X_hat.shape
    print("num_samples, num_times", num_samples,num_times)
    score, score_occupied, score_occluded, score_free = 0, 0, 0, 0
    MS_scores = np.zeros((num_samples, num_times-start_time,3))

    for sample in range(num_samples):
        if sample%50==0:
            print(sample)
        for t in range(start_time, num_times):
            occupied, occluded, free = computeSimilarityMetric(X_test[sample,t,:,:], X_hat[sample,t,:,:])
            score += occupied
            score += occluded
            score += free
            MS_scores[sample, t-start_time,0] = occupied
            MS_scores[sample, t-start_time,1] = occluded
            MS_scores[sample, t-start_time,2] = free

    avg_score = score/(num_samples*(num_times-start_time))

    return avg_score, MS_scores

def toDiscrete(m):
    """
    Args:
        - m (m,n) : np.array with the occupancy grid
    Returns:
        - discrete_m : thresholded m
    """

    y_size, x_size = m.shape
    m_occupied = np.zeros(m.shape)
    m_free = np.zeros(m.shape)
    m_occluded = np.zeros(m.shape)

    #Handpicked
    occupied_value = 0.85
    occluded_value = 0.20

    for x in range(x_size):
        for y in range(y_size):
            val = m[y,x]

            if val >= occupied_value:
                m_occupied[y,x] = 1
            elif val >= occluded_value:
                m_occluded[y,x] = 1
            else:
                m_free[y,x] = 1

    return m_occupied, m_occluded, m_free

def todMap(m):

    """
    Extra if statements are for edge cases.
    """


    y_size, x_size = m.shape
    dMap = np.ones(m.shape) * np.Inf

    for y in range(y_size):
        for x in range(x_size):
            if m[y,x] == 1:
                dMap[y,x] = 0


    for y in range(0,y_size):
        if y == 0:
            for x in range(1,x_size):
                h = dMap[y,x-1]+1
                dMap[y,x] = min(dMap[y,x], h)

        else:
            for x in range(0,x_size):
                if x == 0:
                    h = dMap[y-1,x]+1
                    dMap[y,x] = min(dMap[y,x], h)
                else:
                    h = min(dMap[y,x-1]+1, dMap[y-1,x]+1)
                    dMap[y,x] = min(dMap[y,x], h)

    for y in range(y_size-1,-1,-1):

        if y == y_size-1:
            for x in range(x_size-2,-1,-1):
                h = dMap[y,x+1]+1
                dMap[y,x] = min(dMap[y,x], h)

        else:
            for x in range(x_size-1,-1,-1):
                if x == x_size-1:
                    h = dMap[y+1,x]+1
                    dMap[y,x] = min(dMap[y,x], h)
                else:
                    h = min(dMap[y+1,x]+1, dMap[y,x+1]+1)
                    dMap[y,x] = min(dMap[y,x], h)

    return dMap

def computeDistance(m1,m2):

    y_size, x_size = m1.shape
    dMap = todMap(m2)
    d = 0
    num_cells = 0
    for y in range(0,y_size):
        for x in range(0,x_size):
            if m1[y,x] == 1:
                num_cells += 1
                d += dMap[y,x]

    if num_cells != 0:
        output = d/num_cells

    if num_cells == 0 or d == np.Inf:
        output = y_size + x_size

    return output

def computeSimilarityMetric(m1, m2):

    m1_occupied, m1_occluded, m1_free = toDiscrete(m1)
    m2_occupied, m2_occluded, m2_free = toDiscrete(m2)

    occluded = computeDistance(m2_occluded, m1_occluded) + computeDistance(m1_occluded, m2_occluded)
    occupied = computeDistance(m1_occupied,m2_occupied) + computeDistance(m2_occupied,m1_occupied)
    free = computeDistance(m1_free,m2_free) + computeDistance(m2_free,m1_free)

    return occupied, occluded, free