# Introduction to Machine Learining Basics

## Introduction

* __When__ and __Why__ to use Machine Learning
* paradigms
* fundamental Ingredients
* statistical Learning Theory

## When

__When__ the considered system should

* __adapt__ to the surrounding environment
* __improve__ its performance with respect to a specific computational task
* __discover__ regularities and new information from empirical data
* __acquire__ new computational capabilities

## Why

__Why__ not to use traditional algorithmic approach

* impossible to exactly formalize the problem solved
* presence of noise and/or uncertainty
* high complexity in formulating a solution
* lack of compiled knowledge with respect to the problem to be solved

## Data

### Tipically

* data is available
  * obtained once for all
  * acquired incrementally by interacting with the environment
* knowledge of the application domain is available, however
  * incomplete
  * imprecise

### Desiderata

to use data for

* acquiring new knowledge
* refining the already available knowledge
* correcting the already available knowledge

### Example - Handwritten Digit Recognition

![handwritten digits](./Immagini/digits.png)

* impossible to exactly formalize the problem (only examples are available)
* noise may be present and data may present ambiguities

## Lines of Research within Machine Learning

* Induction of Rules/Decision Trees from data
* Neural Networks
* Instance Based Learning
* Probabilistic (Bayesian) learning
* Reinforcement learning
* Genetic Algorithms
* Inductive Logic Programming

## Main Learning Paradigms

### Supervised Learning

* given pre-classified examples learn a general description which captures the information content of the examples
* it should be possible to use this description in a predictive way
* it is assumed that an expert provides the supervision

### Unsupervised Learning

* given a set of examples discover regularities and/or patterns
* there si no expert to help us

### Reinforced Learning

* __agent__ which may
  * be in _state s_, and
  * execute an _action a_
* and operates in an _environment e_, which in response to action _a_ in state _s_ returns
  * the _next state_, and
  * a _reward r_, which can be _positive_, _negative_ or _neutral_

The goal of the agent is to maximize a function of the rewards

## Fundamental Ingredients

* __Training Data__ (Drawn from the __Instance Space X__)
* __Hypotesis Space, H__
  * it constitutes the set of functions which can be implemented by the machine learning system
  * it is assumed that the function to be learned _f_ may be represented by a hypothesis _h in H_
  * or that at least a hypothesis _h in H_ is "similar" to _f_
* Search Algorithm into the Hypothesis Space, __Learning Algorithm__

__WARNING:__ _H_ cannot coincide with the set of all possible functions and the search to be exhaustive

__[Inductive Bias](https://en.wikipedia.org/wiki/Inductive_bias):__ The group of assumptions used on the representation and/or the search

## Example of hypothesis space

Hyperplanes in R^2

* Instance Space -> points into the plane
* Hypothesis Space -> dichotomies induced by hyperplanes in R^2

![dichotomies inducted by hyperplanes in R^2](./Immagini/dichotomies.png)

## Example of Inductive Bias in Concept Learning

__Definition:__ A concept on an Instance Space __X__ is defined as a boolean function on _X_.

__Definition:__ An example of a concept _c_ on the Instance Space _X_ is defined as a couple _(x,c(x))_, where _x in X_ and _c()_ is a boolean function.

__Definition:__ Let _h_ be a boolean function defined in the Instance Space _X_. We say that _h_ satisfies _x in X_ if _h(x)=1_ (True).

__Definition:__ Let _h_ be a boolean function defined on the Instance Space _X_ and _(x,c(x))_ an example of _c()_. We say that _h_ is consistent  with the example if _h(x) = c(x)_. Moreover we say that _h_ is consistent with a set of examples _Tr_ if _h_ is consistent with every example in _Tr_.

## Hypothesis Space: conjunctions of literals

[Conjunction of _m_ literals](https://en.wikipedia.org/wiki/Conjunctive_normal_form)

* Instance Space -> strings of _m_ bits: _X = {s|s in {0,1}^m}_
* Hypotesis Space -> all the logic sentences involving literals and just containing the operator ∧:

  ![conjunction of literals](./Immagini/conjunction_of_literals.png)

Notice that if in a formula a literal occurs together with its negation, then the formula is always _false_. So, all the formulas containing a literal and its negation, are equivalent to _false_.

![conjunction of literals e.g.](./Immagini/conjunction_of_literals_eg.png)

## Hypotesis Space of Boolean Functions

Conjunction of _m_ literals

How many distincts hypothesis there are as a function of _m_?

Considering that all the unsatisfiable formulas are equivalent to false, we do not consider formulas where a literal occurs together with its negation.
So, for each possible bit of the input string the corresponding literal may not be present in the logic formula or, if it appears, it is either asserted or negated:

3\*3\*3\*3...3 = 3^m

And considering the always false formula, we get 3^m + 1

## Hypotesis space: partial order

![partial order](./Immagini/partial_order.png)

## Learning Conjunctions of Literals

__Find S__ Algorithm
/* it finds the most specific hypothesis which is consistent with the training set */

* input: training set _Tr_
* initiate _h_ to the most specific

  ![literals hypothesys](./Immagini/literals_hypothesis.png)
* for each positive training instance (_x_, _true_) in _Tr_
  * remove from _h_ any literal which is not satisfied by _x_
* returns _h_

## Example of application: _m_=5

![example of application](./Immagini/example_of_application.png)

## Inductive Bias

![inductive bias](./Immagini/inductive_bias.png)

## Hypothesis Space of Boolean Functions

Lookup Table

* Instance Space -> strings of _m_ bits: _X = {s|s in {0,1}^m}
* Hypothesis Space -> all the possible truth tables which map input instances to the _true_ and _false_: _H = {f(s)|f : X ->{true,false}}_

  ![lookup table](./Immagini/lookup_table.png)

How many distinct hypothesis there are as a function of _m_?

By a lookup table it is possible to implement any boolean function on the Instance Space

Since the number of possible instances is:

2\*2\*2\*2...2 = 2^m

the number of distinct funtions is: 2^(2^m)

## Another Example of Learning Algorithm

### Training algorithm for a Perceptron

input: training set _Tr = {(x,t)}_, where _t in {-1,+1}_

1. inizialize the weight vector w to the null vector (all components equal to 0)
2. repeat
    1. select (at random) one training example _(x,t)_
    2. if _out = sign(w*x) != t_ then

        w<-w+(t-out)x

![geometric interpretation](./Immagini/perceptron.png)

### Example of Execution

![perceptron execution](./Immagini/perceptron_execution.png)
![perceptron execution](./Immagini/perceptron_execution_2.png)