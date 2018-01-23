# coding: utf-8

def generate_simulations (mean1, mean2, sd, n, n_simulations):
    '''This function generates a distribution of p-values coming from to
    samples, sampled from a normal distribution with mean and standard deviation
    selected by the corresponding parameters.'''


    import numpy as np
    import scipy.stats

    p_values = np.zeros(n_simulations) # Variable that will store the p-values
    for i in range(n_simulations): # Generate random values for each group
        values_group1 = np.random.normal(loc = mean1, scale = sd, size = n)
        values_group2 = np.random.normal(loc = mean2, scale = sd, size = n)

        # Calculate a t test for independent samples,
        # comparing the values of each group
        t_value, p_values[i] = scipy.stats.ttest_ind(values_group1, values_group2)

    return p_values


def plot_simulation (p_values0, p_values1, bins = 100, truncated = False):
    '''Plot the histogram of two distributions of p-values'''
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(dpi = 120)
    # Plot the density of the distribution of p-values
    weighted = np.ones_like(p_values0)/float(len(p_values1))
    hist_vals = plt.hist([p_values0, p_values1], color=['r','b'],
                        bins = bins, weights = [weighted, weighted],
                        histtype = 'step')
    # Change the x axis ticks and labels for a better visualization
    plt.xticks([i/20 for i in range(21)], rotation = 'vertical')
    # Reduce the maximum value of y axis if requested
    if truncated:
        plt.ylim(0, 0.1)
        
    # Include the axis labels and legend
    plt.xlabel('p-values', fontsize = 'large')
    plt.ylabel('density', fontsize = 'large')
    plt.legend(labels = ['H0','H1'])
    
    return hist_vals[0]


def plot_lines_likelihood(values, truncated = False):
    '''Plots two distributions of p-values, and prints the likelihood of an
    interval of p-values comparing the two distributions.'''
    import matplotlib.pyplot as plt

    plt.figure(dpi = 120)
    #Plot the two lines
    plt.plot(values[0], color = 'b')
    plt.plot(values[1], color = 'r')
    # Change the x axis ticks and labels for a better visualization
    temp = plt.xticks(range(0,101,5), tuple([str(i/20) for i in range(21)]),
                    rotation = 'vertical')
    # Reduce the maximum value of y axis if requested
    if truncated:
        plt.ylim(0, 0.1)

    # Include the axis labels and legend
    plt.xlabel('p-values', fontsize = 'large')
    plt.ylabel('density', fontsize = 'large')
    plt.legend(labels = ['H0','H1'])
    
    #Calculate and print the likelihood of an interval of p-values under H0 and H1
    likelihood = sum(values[0][4:6])/sum(values[1][4:6])
    print(('Likelihood of interval of p-values from 0.03 to 0.05:' +
        '\n{:0.2f} times more likely when H0 is true than when H1 is' +
        'true').format(likelihood))


def plot_lines(values, truncated = True):
    '''Plots two distributions of p-values, and prints the likelihood of an
    interval of p-values comparing the two distributions.'''
    import matplotlib.pyplot as plt

    plt.figure(dpi = 120)
    #Plot the two lines
    plt.plot(get_values_hist(values), color = 'b')
    # Change the x axis ticks and labels for a better visualization
    temp = plt.xticks(range(0,101,5), tuple([str(i/20) for i in range(21)]),
                    rotation = 'vertical')
    # Reduce the maximum value of y axis if requested
    if truncated:
        plt.ylim(0, 0.1)

    # Include the axis labels and legend
    plt.xlabel('p-values', fontsize = 'large')
    plt.ylabel('density', fontsize = 'large')
    plt.legend(labels = ['H0'])


def get_values_hist(p_values0):
    import numpy as np
    weighted = np.ones_like(p_values0)/float(len(p_values0))
    return np.histogram(p_values0, bins = 100, weights = weighted)[0]

