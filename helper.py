from multiprocessing.connection import wait
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, avg_scores):
    display.clear_output(wait = True)

    display.display(plt.gcf())  

    plt.clf()

    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores)
    plt.plot(avg_scores)

    plt.ylim(ymin = 0)
    plt.text(len(scores)-1, scores[-1], 'Score: %d' % scores[-1])
    plt.text(len(scores)-1, avg_scores[-1], 'Average: %d' % avg_scores[-1])

    plt.show(block = False)
    plt.pause(0.1)
