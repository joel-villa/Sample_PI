# Sample Power Iteration

Speeding up Power Iteration via a sampling scheme based on row magnitudes. This
approach has theoretical backing

## The Object Oriented Design

- Main objects: Tests, Results, MatGetter
- These two objects guide the structuring of this testing suite

## Using `Ruff`

Ruff is a linter (error checking) and style checker 

- Install with: `pip install ruff`
- Lint with: `ruff check`
- Style checking: `ruff format`
- *All in one command*: `ruff --fix`

## A Note on ssgetpy

- On UNIX based machines, the ssgetpy library will download matrices onto your 
machine, at the root in the .ssgetpy directory
- Some of the matrices involved in this test suite are quite large, run 
  scripts with caution 

## TODOs

### Empirical

- [ ] Have result able to calculate work itself
- [ ] Work include row norms cost (dot product => nnz scalar mults)
- [ ] Run tests (all three types)
- [ ] Dense matrices 
  - Row powerlaw? PSD?
  - A vs. A^TA

### Theory

- [ ] Prove first moment is not much worse than 2nd moment
- [ ] Expected amount of reduced work per iteration (expected new nnzs)
- [ ] Binomial expected work: (1 - ||a_j||)^N
- [ ] FAST-PI requires $O(\dots)$ scalar mults, vs. $O(\dots)$ of the standard 
  power-iteration...
