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

- [ ] FIGURE OUT WHAT MAKES SOME PERFORMANT AND OTHERS NOT

### Empirical

- [ ] Doc comments are definitely outdated
- [ ] First moment average (plus proof to come)
- [x] Finish objectifying things (*man noises*)
- [ ] Move Work calculator inside of results :D
- [ ] Test on dense matrices (are there power-law drop off dense matrices)
- [ ] Do norms impact performance? Why are some not seeing gains?
- [ ] Run on CARC:
  - [ ] Collection test sweep averaging (Ask Saia about this: is it possible if 
    we have variable number of iteratiosn?)
    - [ ] Solution: do 20 randomly selected mats dense & sparse alike
  - [ ] Run on large matrices!
- [ ] Convergence vs. Work graph + line for bounds on $(1 \pm \epsilon)||A||$
- [x] Make code strongly typed and redo doc comments
- [x] Move two_norm.npz handling into proven/util/npz_wrapper.py

### Theory

- [ ] Prove first moment is not much worse than 2nd moment
- [ ] Expected amount of reduced work per iteration (expected new nnzs)
- [ ] Binomial expected work: (1 - ||a_j||)^N
- [ ] Take advantage of eigen-gap? Other routes forward?
- [ ] FAST-PI requires $O(\dots)$ scalar mults, vs. $O(\dots)$ of the standard 
  power-iteration...
