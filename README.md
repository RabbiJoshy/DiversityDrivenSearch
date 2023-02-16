# DiversityDrivenSearch
ZMART Algorithm - Zoektocht (met) Meer Afwisseling (en een) Rendement Toename

A project I completed professionally for a real-estate develoment software company. The software optimizes building development by generating thousands of potential building design variants subject to developer constraints. 

To ensure that each variant is functionally valid, computationally expensive 3D rendering and engineering studies are performed that test whether a variant meets government standards for minimum daylight and maximum noise levelsas well as other studies on wind modelling and thermal efficiency.

Instead of generating every single variant at once, the algorithm uses a small subset of the variants to model the multi-dimensional outspace and iteratively suggests batches to search, that will maximise diversity of output solution and subject to a range of potential constraining output values.

Diversity in this case is defined using PCA to reduce the outputs into a low dimensional space. 

Therefore on each batch, the variants chosen to go through the engineering studies are those that are predicted to maximise distance from all other computed points in the reduced space.

If a client wants to see 1,000,000 variants, but each computation takes 8.5 seconds, then this will take 100 days to run. 
If only 14 days are available, Using this ZMART algorithm, the optimal 14% of solutions can be chosen subject to both maximising diversity and constraints on values of the output such as cost, light quality and ground floor space.
