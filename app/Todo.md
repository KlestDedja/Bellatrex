# ToDo list

Here we list ideas to be implemented and keep track of issues that arise and need a fix.

Issues are grouped by the estimated amount of work (and not by priority). Contributors that are willing to

## Major items

- Code testing with ``pytest``, it's about time!

- Introduce ``$\gamma$`` (or other name) parameter to include leaf predictions to the (weighted) representation of the trees. The resulting representation would have $d+1$ dimensions, as opposed to the current $d$.

## Minor items

- Update documentation! In hindsight, many things are still unclear

- For multi-output tasks, enable users to:

    - be able to select a subset of targets to run explanations for;

    - be able to select a (single) target to run ``plot_visuals()``

