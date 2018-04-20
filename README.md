# Path-Finder
Implementation of a deep reinforcement learning model trained to solve pathfinder puzzles.

See results commit on 4-22-17.  Observation: convergence and convergence rate are relatively 
unphased by maximum number of actions except when the number is insufficient.

5/28
todo

- implement scheduler:
    1. learning rate scheduler
    (1.5. MNA scheduler)
    2. state presentation scheduler
- implement deuling architectures (easy modification)

2/7/18
Dear reader, these are the steps to run the program:
- {{source [venv]/bin/activate}}: this env ought to have python (unsure whether developed on/for python2.7 or python3.5) with tensorflow installed.
- {{python experiment.py X}} for the most automated experiment launch, where X is one of <ego-allo-basic> or <ego-allo-forked>.
- the two files analysis and save_as_plot also do some essential operations on offline/completed experiments.

4/18/18
- from directory ../forked_openai_baselines, imported launch_experiment.py to this directory (Path-Finder) for openai-free implementation.
Todo: consolodate all the static resource files into a single file, then begin rewriting exp

question: when/where to populate network parameters given keywork [network]?
