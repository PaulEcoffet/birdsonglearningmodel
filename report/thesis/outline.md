Introduction
===============
## Zebra Finch song learning
### Characteristic of zebra finch song learning
 * Learn one song for its whole life, Close-end learner (opposition open-end learer)
 * Learn the song of father
 * Song divided in motifs, syllables and notes
 * subsong (babbling), plastic song, cristallysation
 * Sensory phase (memorisation of tutor song), sensory motor phase
   * If no auditory feedback, cannot learn.

### Why is zebra finch song learning studied
 * Model of Human speech learning
     * Actual song learning, not innate
     * Song with complex structures
 * Well studied Neuroanatomy
 * Easy to study experimentally
     * Easily domesticated
     * Learn one song
     * Learn quickly (90DPH)
     * Easy to track song development

## Neurobiology of the Zebra Finch
### Neuroanatomy of the Zebra Finch song system
 * Connection between RA, HVC, Area X, … Inhibition, excitation

### Pattern of activation in RA and HVC
 * HVC clock like, temporal structure (Ali et al.)
 * RA activation while singing at very precise time and sparse coding
    * Motor control (Ali et al.)
 Ali et al. shows real two different learning: spectral and temporal

## Models of song learning
Only very few models have been created. Even less are actual computational models.

### Reinforcement learning
* Proposed but no real explanation of what could be the state space, the action space, the reward function (Dave&Margoliash).
* Used in paradigm to test different hypothesis (averse reward to force change in behaviour of the bird)

### Song preferences in selection (Marler)
* Behavioural model to explain how the bird select its template
* TODO: Add more

### Coen's model
* Clustering technique with babbling (multimodal)
     * Cluster the tutor song syllables thanks to their characteristics
     * Babbling, create a mapping between the motor space and the identified cluster
* Use of a real synthesizer but not actually built to model zf vocal apparatus
* No quantitative means to see how good is the song reproduction
* The learning is only babbling, nothing is driving the model in a specific direction.

## Song synthesizer
### Description of Perl song synthesizer to reproduce Zebra Finch song
* Presentation of anatomy simulation, mass and spring…
* Parameters
     * Air sac pressure
     * Syringeal Labial Tension
* Parameters are close to actual motor actions, so close to actual motor command

### Zebra Finches are sensible to song produced by the synthesizer
* Show results of Amador where RA neurons were activated by Synth song but not by conspecific song.

### Gestures and song structure
* Boari's Gesture concept and automatic extraction of the gestures
* Could have been correlated to HVC activation but in fact no.

## Influence of Sleep in the Zebra Finch song development
### Margoliash results with song replay
* RA neurons activated while singing
* Also activated when the bird is asleep and listen to his own song
* Spontaneous activity with part of the song: Replays
* Replays can be consolidation of memory
* Replays can have another role

### Derégnaucourt results about positive impact of sleep for development
* Extraction of syllables characteristics and track over time
* Global trend for the trajectory of a syllable over time
* Each day, the syllables characteristics are closer

## A computational model of birdsong learning to explain the sleep influence
### Interest of a computational model of birdsong learning
* Computational model helps understanding what are the *implementation constraints* of the learning mechanisms
     * Use of synthesizer
     * Realistic computational budget
* Easily make hypotheses that can be tested experimentally afterwards
* Abstracted and controlled environment

### Goal: Build a modular two-step learning model and look for learning algorithm that can account for Derégnaucourt's results.

Our Model
=============
## Global Architecture
### Usage of Boari's implementation of the birdsong synthesizer
### Measurement of song quality with standard measures
* Entropy, Pitch, Goodness, Amplitude, Frequency Modulation, Amplitude Modulation
* Imported from Matlab implementation, with qualitatively similar results

### Two-step learning model
* Bird has several song models it trains to reach tutor
* tutor song is known
* day algorithm for parameters optimisation
* night algorithm for structure optimisation
* Hypothesis: structure optimisation yield unlearning short term, better learning long term

## Song Model
### Song Model
### Gesture paradigm inherited from synthesizer
### Song structure
* List of gestures and their duration
* Fixed duration of the song because of measurement

### Gesture composed of two generators for the motor commands
* Abstracted in sum of sin & linear func

## Day learning algorithm
### goal
 * Optimise gestures parameters

### Hillclimbing
* really simple
* Choose song model, choose gesture
* Choose close parameters, if better keep, if worse trash
* Knows if better by comparison of weighted standard measurements
* Not whole song but only gesture trained to make faster computations
    * Actually creates unlearning

### Prediction
* Should improve song production but get stuck in local maximum because bad structure

## Night learning algorithms
### Goal
* Find better structure to describe song motor command

### Several variations of algorithm have been tested
* Evolutionary algorithm
    * Simple solution for structure variation
* with or without diversity

### Algorithm
* Evolutionary algorithm Microbial GA
* Increase population size and add variation in structure
    * Remove, add, change, copy gesture
    * Song always the same length for comparison reasons.
* Compare by tournament
     * The winner put a variation of itself in place of the loser
     * Compare number of neighbour * score, lower the better

### Predictions
 * Structure variation yields unlearning short term but positive impact long
  term
 * Diversity will increase this

## Parameters
* Tried to be realistic
* most are fit through gridsearch
* Realistics: Number of days, number of syllables sung during all dev
* Gridsearch optimisation
* Default value for gesture parameters
* Learning rate
     * Prevent part of unlearing
     * Could be fitted to match real song learning rate
     * Coefficient for score optimisation
* Algorithm way better in score than Boari but qualitatively very different to the ear
* Look at which parameters boari's method was better than algo and put priority on them
* Amplitude and entropy
* Diversity threshold to maximise variance in diversity score
    * Value: 5000
    * Other parameters
* Number of song models during day and night: Depend of runs
* Boundaries for parameters values: Fixed
* Number of tournaments during night: depend of runs
    * Correlated with replay? By how much?

Analyses and results
=========================
## Learning method is as good Boari's method or better
* Using standard measure criteria in the birdsong community
* Simple description of motor params sufficient to produce good songs
* Qualitatively same amount of gestures
    * Can be due to luck

## Too little training per model cause divergence
* maybe due to global vs local error

## Derégnaucourt results not reproduced
* Syllables extracted by time of begin and end
* Without or with diversity
* No night deterioration
* Night deterioration has no impact in overall learning

Discussion
==============
## The synthesizer which cannot produce every sounds
* Our score really close to boari's method (not way better or way worst), maybe we reached synthesizer limits

## The parameters description we choose
* more simple/complexe possible than sum of sin and affine?

## The unlearning during day due to the gesture learning
## Fixed duration of songs in learning
* Dynamic Time Warping can correct that

## Big artificial separation between structuration and gestures optimisation
## Diversity not strong enough? What if only diversity during night?
* Maybe not convergence
* Maybe what we are looking for

Conclusion
=============
## Learning algorithm with two step learning
* Very few of them
* Working with realistic synthesizer
* modular architecture, easy to test new models

## Restructuration didn't yield the expected effect
* More parameters search might be able to fix it
