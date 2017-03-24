# Zebra finch song learning model


The Zebra Finch Bird Song Learning Model project aims to model the loss in performance of zebra finches after sleep which correlates with an increase in song reproduction at the end of their development. We believe this is due because the zebra finch restructures its knowledge during its sleep.

This project is done during my final internship for the Cogmaster in the AMAC team at the Institut des Systèmes Intelligents et de Robotique in Paris. This project is part of the [DREAM project][1].

## Architecture

The architecture of this model is a two part algorithm: One is a day part, and the second one is a night part. These parts do different things. The day part is an optimisation algorithm, whereas the night part is a restructuration algorithm.

Several model of learning for day and for night have been implemented.

- [The list of day learning models][DLML]
- [The list of night learning models][NDML] 


## Notebook

You can follow my [research notebook](https://osf.io/ja8k9/wiki/Notebook/) to see the progression of my work.

[1]: http://robotsthatdream.eu
[DLML]: https://osf.io/ja8k9/wiki/Day%20Learning%20Models%20List/
[NDML]: https://osf.io/ja8k9/wiki/Night%20Learning%20Models%20List/
